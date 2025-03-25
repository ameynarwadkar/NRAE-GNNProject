import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import argparse
import yaml
from omegaconf import OmegaConf
from loader import get_dataset, get_dataloader
from models import get_model

from PIL import Image

def standardize_frame(frame, size=(480, 640)):
    """Resize and convert to RGB"""
    if frame.ndim == 2:  # grayscale image
        frame = np.stack([frame]*3, axis=-1)
    elif frame.shape[2] == 1:  # grayscale channel
        frame = np.repeat(frame, 3, axis=2)
    img = Image.fromarray(frame)
    img = img.resize(size)
    return np.array(img)

def plot_loss_curves(epochs, train_losses, test_losses, logdir, model_name):
    """
    Plots and saves training and testing loss curves over epochs.
    
    Parameters:
        epochs (array-like): Epoch indices.
        train_losses (array-like): Training loss per epoch.
        test_losses (array-like): Testing loss per epoch.
        logdir (str): Directory to save the figure.
        model_name (str): Name of the model (e.g., "AE" or "NRAE").
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label=f"{model_name} Training Loss", marker='o')
    plt.plot(epochs, test_losses, label=f"{model_name} Testing Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Testing Loss vs Epochs ({model_name})")
    plt.legend()
    plt.grid(True)
    loss_fig_path = os.path.join(logdir, f"{model_name}_loss_curves.png")
    plt.savefig(loss_fig_path)
    plt.show()
    print(f"Loss curves saved to: {loss_fig_path}")

def run(cfg):
    # Setup seeds
    seed = cfg.get("seed", 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup device
    device = cfg.device

    # Setup Dataloader
    d_datasets = {}
    d_dataloaders = {}
    for key, dataloader_cfg in cfg["data"].items():
        d_datasets[key] = get_dataset(dataloader_cfg)
        d_dataloaders[key] = get_dataloader(dataloader_cfg)

    # Setup Model
    model = get_model(cfg['model']).to(device)
    model_name = cfg['model']['arch'].upper()
    if cfg["data"]["training"].get("graph", False):
        model.dist_indices = d_datasets['training'].dist_mat_indices

    # Setup optimizer and scheduler
    params = {k: v for k, v in cfg['optimizer'].items() if k != "name"}
    optimizer = torch.optim.Adam(model.parameters(), **params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Lists to record loss per epoch
    all_train_losses = []
    all_test_losses = []

    num_epochs = cfg['training']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        training_loss = []
        if cfg["data"]["training"].get("graph", False):
            for x, x_nn in d_dataloaders['training']:
                train_dict = model.train_step(x.to(device), x_nn.to(device), optimizer)
                training_loss.append(train_dict["loss"])
        else:
            for x in d_dataloaders['training']:
                train_dict = model.train_step(x.to(device), optimizer)
                training_loss.append(train_dict["loss"])
        avg_train_loss = sum(training_loss) / len(training_loss)
        all_train_losses.append(avg_train_loss)
        print(f"Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}")

        # Compute test loss
        model.eval()
        test_loss = []
        with torch.no_grad():
            for x in d_dataloaders['test']:
                if hasattr(model, "test_step"):
                    out_dict = model.test_step(x.to(device))
                    loss = out_dict["loss"]
                else:
                    out = model(x.to(device))
                    if isinstance(out, dict):
                    # Here you may need to adjust the loss computation to match your model’s output.
                        loss = out_dict.get("loss", 0.0)
                    else:
                        loss = out
                    if torch.is_tensor(loss):
                        loss = loss.mean().item()
                test_loss.append(loss)
        avg_test_loss = sum(test_loss) / len(test_loss) if len(test_loss) > 0 else 0.0
        all_test_losses.append(avg_test_loss)
        print(f"Epoch: {epoch}, Testing Loss: {avg_test_loss:.4f}")

        if epoch > 0.8 * num_epochs:
            scheduler.step()

    # Plot and save loss curves
    epochs = np.arange(1, num_epochs + 1)
    plot_loss_curves(epochs, all_train_losses, all_test_losses, cfg['logdir'], model_name)

def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return float(val)
    except ValueError:
        return val

def parse_unknown_args(l_args):
    n_args = len(l_args) // 2
    kwargs = {}
    for i in range(n_args):
        key = l_args[i * 2]
        val = l_args[i * 2 + 1]
        assert "=" not in key, "optional arguments should be separated by space"
        kwargs[key.strip("-")] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split(".")
        d = d_new_cfg
        for i, k in enumerate(l_key):
            if i == len(l_key) - 1:
                d[k] = val
            else:
                if k not in d:
                    d[k] = {}
                d = d[k]
    return d_new_cfg

def save_yaml(filename, text):
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", default='results/')
    parser.add_argument("--device", default=0)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    if args.device == "cpu":
        cfg["device"] = "cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"

    exp_name = cfg.get("exp_name", "exp")
    cfg['logdir'] = os.path.join(args.logdir, exp_name)
    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg['logdir'], exist_ok=True)
    copied_yml = os.path.join(cfg['logdir'], os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml}")

    run(cfg)
