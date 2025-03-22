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
    if cfg["data"]["training"].get("graph", False):
        model.dist_indices = d_datasets['training'].dist_mat_indices

    # Latent visualization before training
    try:
        training_data_fig = None
        if "image" in cfg["model"]["encoder"]["arch"]:
            training_data_fig = model.image_manifold_visualize(
                epoch=0,
                training_loss=0.0,
                training_data=d_datasets['training'].data,
                labels=getattr(d_datasets['training'], "labels", None),
                device=device,
                title="Initial Latent Space"
            )
        else:
            training_data_fig = model.synthetic_visualize(
                0,
                0.0,
                d_datasets['training'].data,
                d_datasets['test'].data,
                device
            )
    except Exception as e:
        print("Skipping initial visualization due to:", e)
        training_data_fig = None

    # Graph visualization
    graph_fig = None
    if hasattr(d_datasets['training'], 'visualize_graph'):
        try:
            graph_fig = d_datasets['training'].visualize_graph(
                d_datasets['training'].data,
                d_datasets['test'].data,
                d_datasets['training'].dist_mat_indices,
                model=model,
                device=device
            )
        except Exception as e:
            print("Skipping graph visualization:", e)

    # Setup optimizer and scheduler
    params = {k: v for k, v in cfg['optimizer'].items() if k != "name"}
    optimizer = torch.optim.Adam(model.parameters(), **params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    list_of_images = []
    for epoch in range(cfg['training']['num_epochs']):
        training_loss = []
        if cfg["data"]["training"].get("graph", False):
            for x, x_nn in d_dataloaders['training']:
                train_dict = model.train_step(x.to(device), x_nn.to(device), optimizer)
                training_loss.append(train_dict["loss"])
        else:
            for x in d_dataloaders['training']:
                train_dict = model.train_step(x.to(device), optimizer)
                training_loss.append(train_dict["loss"])

        print(f"n_epoch: {epoch}, training_loss: {sum(training_loss)/len(training_loss)}")

        if epoch > 0.8 * cfg['training']['num_epochs']:
            scheduler.step()

        if (epoch % 40 == 0) or (epoch < 30):
            try:
                if "image" in cfg["model"]["encoder"]["arch"]:
                    image_array = model.image_manifold_visualize(
                        epoch,
                        sum(training_loss)/len(training_loss),
                        d_datasets['training'].data,
                        getattr(d_datasets['training'], "labels", None),
                        device=device,
                        title=f"Latent Manifold (Epoch {epoch})"
                    )
                else:
                    image_array = model.synthetic_visualize(
                        epoch,
                        sum(training_loss)/len(training_loss),
                        d_datasets['training'].data,
                        d_datasets['test'].data,
                        device
                    )
                list_of_images.append(image_array)
            except Exception as e:
                print(f"Skipping visualization at epoch {epoch} due to:", e)

    # Convert Images to GIF
    f = plt.figure()
    model_name = cfg['model']['arch'].upper()
    plt.text(0.5, 0.5, f'{model_name} Training', size=24, ha='center', va='center')
    plt.axis('off')
    f.canvas.draw()
    f_arr = np.array(f.canvas.renderer._renderer)
    plt.close()

    list_figs = [f_arr]*10
    if training_data_fig is not None:
        list_figs += [training_data_fig]*10
    if graph_fig is not None:
        list_figs += [graph_fig]*10
    list_figs += list_of_images + [list_of_images[-1]]*20

    for i, img in enumerate(list_figs):
        print(f"Frame {i} shape: {img.shape}")

    standard_size = (480, 640)  # or any consistent resolution

    standardized_figs = [standardize_frame(f, size=standard_size) for f in list_figs]

    imageio.mimsave(
        os.path.join(cfg['logdir'], f'{model_name}_training.gif'),
        standardized_figs,
        duration=1.0
    )

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
