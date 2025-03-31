"""
MNIST training code with additional latent geometry visualizations.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import argparse
import yaml
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from loader import get_dataset, get_dataloader
from models import get_model

def compute_mse_and_psnr(x, recon):
    """
    Computes the Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
    for a batch of original images x and reconstructed images recon.
    
    Assumes that x and recon are torch tensors with pixel values in the range [0, 1].
    """
    mse = F.mse_loss(recon, x).item()
    psnr = 10 * math.log10(1.0 / mse)
    return mse, psnr

def plot_loss_curves(epochs, train_losses, test_losses, logdir, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label=f"{model_name} Training Loss", marker='o')
    plt.plot(epochs, test_losses, label=f"{model_name} Testing Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Epochs ({model_name})")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(logdir, f"{model_name}_loss_curves.png")
    plt.savefig(loss_path)
    plt.show()
    print(f"Loss curves saved to: {loss_path}")

def visualize_scalar_curvature_field(model, device, grid_range=(-3, 3), grid_size=50, save_path=None):
    """
    Computes a proxy scalar curvature field over a 2D grid in the latent space and plots a heatmap.
    This assumes the latent space is 2-dimensional.
    """
    x_vals = torch.linspace(grid_range[0], grid_range[1], grid_size, device=device)
    y_vals = torch.linspace(grid_range[0], grid_range[1], grid_size, device=device)
    curvature_field = np.zeros((grid_size, grid_size))

    def compute_curvature_at_point(model, z):
        def f(z_input):
            z_input = z_input.unsqueeze(0)  # shape (1, 2)
            out = model.decoder(z_input)
            target = torch.zeros_like(out)
            loss = F.mse_loss(out, target)
            return loss
        hess = torch.autograd.functional.hessian(f, z)
        return torch.norm(hess).item()

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z = torch.tensor([x.item(), y.item()], device=device, requires_grad=True)
            curvature_field[i, j] = compute_curvature_at_point(model, z)

    plt.figure(figsize=(8, 6))
    plt.imshow(curvature_field, origin='lower', extent=(grid_range[0], grid_range[1],
                                                        grid_range[0], grid_range[1]), cmap='hot')
    plt.colorbar(label="Curvature")
    plt.title("Scalar Curvature Field over Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Scalar curvature heatmap saved to {save_path}")
    plt.show()

def visualize_manifold_comparison(noisy_data, vanilla_recon, nrae_recon, mode_label="NRAE-Q", save_path=None):
    """
    Creates a side-by-side plot comparing:
      - Noisy training data,
      - The vanilla AE reconstruction, and
      - The NRAE reconstruction.
    Overlays error vectors from noisy data to reconstruction.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    
    axs[0].scatter(noisy_data[:,0], noisy_data[:,1], color='blue', label='Noisy Data')
    axs[0].set_title("Noisy Training Data")
    axs[0].legend()
    
    axs[1].scatter(vanilla_recon[:,0], vanilla_recon[:,1], color='red', label='Vanilla AE')
    for i in range(noisy_data.shape[0]):
        axs[1].arrow(noisy_data[i,0], noisy_data[i,1],
                     vanilla_recon[i,0]-noisy_data[i,0],
                     vanilla_recon[i,1]-noisy_data[i,1],
                     head_width=0.03, head_length=0.05, fc='gray', ec='gray', alpha=0.6)
    axs[1].set_title("Vanilla AE Reconstruction")
    axs[1].legend()
    
    axs[2].scatter(nrae_recon[:,0], nrae_recon[:,1], color='green', label=mode_label)
    for i in range(noisy_data.shape[0]):
        axs[2].arrow(noisy_data[i,0], noisy_data[i,1],
                     nrae_recon[i,0]-noisy_data[i,0],
                     nrae_recon[i,1]-noisy_data[i,1],
                     head_width=0.03, head_length=0.05, fc='gray', ec='gray', alpha=0.6)
    axs[2].set_title(f"{mode_label} Reconstruction")
    axs[2].legend()
    
    plt.suptitle("Manifold Comparison", fontsize=16)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Manifold comparison figure saved to {save_path}")
    plt.show()

def run(cfg):
    # Setup seeds and device
    seed = cfg.get("seed", 1)
    print(f"Running with random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = cfg.device
    
    # Setup Dataloaders
    d_datasets = {}
    d_dataloaders = {}
    for key, dataloader_cfg in cfg["data"].items():
        d_datasets[key] = get_dataset(dataloader_cfg)
        d_dataloaders[key] = get_dataloader(dataloader_cfg)
    
    # Setup Model
    model = get_model(cfg['model']).to(device)
    if cfg["data"]["training"].get("graph", False):
        model.dist_indices = d_datasets['training'].dist_mat_indices
    
    # Setup optimizer and scheduler
    params = {k: v for k, v in cfg['optimizer'].items() if k != "name"}
    optimizer = torch.optim.Adam(model.parameters(), **params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
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
        avg_loss = sum(training_loss) / len(training_loss)
        all_train_losses.append(avg_loss)
        print(f"Epoch {epoch}, Training Loss: {avg_loss:.6f}")
        
        # Compute test loss along with MSE and PSNR
        model.eval()
        test_loss = []
        mse_total = 0.0
        psnr_total = 0.0
        num_batches = 0
        with torch.no_grad():
            for x in d_dataloaders['test']:
                x = x.to(device)
                recon = model(x)
                mse_batch, psnr_batch = compute_mse_and_psnr(x, recon)
                mse_total += mse_batch
                psnr_total += psnr_batch
                num_batches += 1
                if hasattr(model, "test_step"):
                    loss = model.test_step(x)["loss"]
                else:
                    loss = mse_batch
                test_loss.append(loss)
        avg_test = mse_total / num_batches if num_batches > 0 else 0.0
        avg_psnr = psnr_total / num_batches if num_batches > 0 else 0.0
        all_test_losses.append(avg_test)
        print(f"Epoch {epoch}, Testing MSE: {avg_test:.6f}, Testing PSNR: {avg_psnr:.2f} dB")
        
        if epoch > 0.8 * num_epochs:
            scheduler.step()
    
    # Visualize training loss curves
    epochs_arr = np.arange(1, num_epochs + 1)
    plot_loss_curves(epochs_arr, all_train_losses, all_test_losses, cfg['logdir'], cfg['model']['arch'].upper())
    
    # MNIST visualization (assumes model has a .mnist_visualize method)
    if hasattr(model, "mnist_visualize"):
        vis_path = os.path.join(cfg['logdir'], f"{type(model).__name__}.gif")
        model.mnist_visualize(d_datasets['training'].data, device, vis_path)
    
    # New Visualization: Scalar Curvature Field Heatmaps for different num_nn values (if latent dim is 2)
    latent_sample = next(iter(d_dataloaders['training']))[0].to(device)
    with torch.no_grad():
        z_sample = model.encoder(latent_sample)
    if z_sample.shape[1] == 2 and hasattr(model, "decoder"):
        num_nn_list = cfg["data"]["training"]["graph"].get("num_nn_list", [cfg["data"]["training"]["graph"]["num_nn"]])
        heatmaps = []
        grid_range = (-3, 3)
        grid_size = 50
        x_vals = torch.linspace(grid_range[0], grid_range[1], grid_size, device=device)
        y_vals = torch.linspace(grid_range[0], grid_range[1], grid_size, device=device)
        
        for nn in num_nn_list:
            new_train_cfg = OmegaConf.to_container(cfg["data"]["training"], resolve=True)
            new_train_cfg["graph"]["num_nn"] = nn
            ds_new = get_dataset(new_train_cfg)
            model.dist_indices = ds_new.dist_mat_indices
            
            curvature_field = np.zeros((grid_size, grid_size))
            def compute_curvature_at_point(model, z):
                def f(z_input):
                    z_input = z_input.unsqueeze(0)
                    out = model.decoder(z_input)
                    target = torch.zeros_like(out)
                    loss = F.mse_loss(out, target)
                    return loss
                hess = torch.autograd.functional.hessian(f, z)
                return torch.norm(hess).item()
            
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    z = torch.tensor([x.item(), y.item()], device=device, requires_grad=True)
                    curvature_field[i, j] = compute_curvature_at_point(model, z)
            heatmaps.append(curvature_field)
        
        # Create subplots for heatmaps; ensure axes is always a list
        fig, axes = plt.subplots(1, len(heatmaps), figsize=(6 * len(heatmaps), 6))
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for idx, hm in enumerate(heatmaps):
            axes[idx].imshow(hm, origin='lower',
                             extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]),
                             cmap='hot')
            axes[idx].set_title(f"num_nn = {num_nn_list[idx]}")
            axes[idx].set_xlabel("z1")
            axes[idx].set_ylabel("z2")
            fig.colorbar(axes[idx].images[0], ax=axes[idx], label="Curvature")
        plt.suptitle("Scalar Curvature Heatmap")
        heatmap_path = os.path.join(cfg['logdir'], "scalar_curvature_heatmaps.png")
        plt.savefig(heatmap_path)
        plt.show()
        print(f"Scalar curvature heatmaps saved to {heatmap_path}")
    
    # New Synthetic Visualization: Manifold Comparison (dummy data for demonstration)
    synthetic_path = os.path.join(cfg['logdir'], "manifold_comparison.png")
    N = 100
    x_vals_np = np.linspace(-3, 3, N)
    noisy_data = np.vstack([x_vals_np, np.sin(x_vals_np) + np.random.normal(scale=0.2, size=N)]).T
    vanilla_recon = np.vstack([x_vals_np, np.sin(x_vals_np) + np.random.normal(scale=0.5, size=N)]).T
    nrae_recon = np.vstack([x_vals_np, np.sin(x_vals_np)]).T
    visualize_manifold_comparison(noisy_data, vanilla_recon, nrae_recon, mode_label="NRAE-Q", save_path=synthetic_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", default='results/')
    parser.add_argument("--device", default=0)
    args, unknown = parser.parse_known_args()
    
    def parse_arg_type(val):
        if val.isnumeric():
            return int(val)
        if val.lower() in ["true", "false"]:
            return val.lower() == "true"
        try:
            return float(val)
        except ValueError:
            return val

    def parse_unknown_args(l_args):
        n_args = len(l_args) // 2
        kwargs = {}
        for i in range(n_args):
            key = l_args[i*2]
            val = l_args[i*2+1]
            kwargs[key.strip("-")] = parse_arg_type(val)
        return kwargs

    def parse_nested_args(d_cmd_cfg):
        d_new = {}
        for key, val in d_cmd_cfg.items():
            parts = key.split(".")
            d = d_new
            for i, p in enumerate(parts):
                if i == len(parts)-1:
                    d[p] = val
                else:
                    if p not in d:
                        d[p] = {}
                    d = d[p]
        return d_new

    def save_yaml(filename, text):
        with open(filename, "w") as f:
            yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    cfg["device"] = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    cfg['logdir'] = os.path.join(args.logdir, cfg['exp_name'])
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg['logdir'], exist_ok=True)
    copied_yml = os.path.join(cfg['logdir'], os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"Config saved as {copied_yml}")
    
    run(cfg)
