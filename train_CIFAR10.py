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
import torchvision
from torchvision.utils import make_grid, save_image

def standardize_frame(frame, size=(480, 640)):
    """Resize and convert to RGB (if needed)."""
    if frame.ndim == 2:  # grayscale image
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 1:  # single channel
        frame = np.repeat(frame, 3, axis=2)
    img = Image.fromarray(frame)
    img = img.resize(size)
    return np.array(img)

def plot_loss_curves(epochs, train_losses, test_losses, logdir, model_name):
    """
    Plots and saves training and testing loss curves over epochs.
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

def compare_original_and_reconstructions(originals, reconstructions, save_path="comparison.png", nrow=8):
    """
    Creates a single image with the originals in the top row and reconstructions in the bottom row.
    """
    both = torch.cat([originals, reconstructions], dim=0)
    grid = make_grid(both, nrow=nrow, normalize=True, pad_value=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(grid, save_path)
    print(f"Comparison image saved to: {save_path}")
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Top: Original Images\nBottom: Reconstructions")
    plt.show()

def save_rotated_sequence(sample_img, save_path, num_frames=20):
    """
    Given a PIL image sample, rotates it step-by-step and stitches the first num_frames
    rotated images side-by-side into one final image.
    
    Args:
        sample_img (PIL.Image): The CIFAR10 sample image.
        save_path (str): Where to save the final stitched image.
        num_frames (int): Number of rotated frames to generate.
    """
    sample_img = sample_img.convert("RGB")
    width, height = sample_img.size
    step = 360 / num_frames  # full rotation spread evenly over num_frames
    rotated_images = []
    for i in range(num_frames):
        angle = i * step
        rotated = sample_img.rotate(angle, resample=Image.BICUBIC, expand=True)
        rotated = rotated.resize((width, height))
        rotated_images.append(rotated)
    total_width = width * num_frames
    new_img = Image.new("RGB", (total_width, height))
    for i, img in enumerate(rotated_images):
        new_img.paste(img, (i * width, 0))
    new_img.save(save_path)
    print(f"Rotated sequence saved to: {save_path}")

##############################################
# New helper function to compare standard and neighborhood reconstructions
##############################################
def compare_standard_and_neighborhood_recon(model, x, x_nn, device):
    """
    Compute and return both standard and neighborhood reconstructions for a batch.
    
    Args:
        model: The CIFAR10AE model.
        x: A batch of central images, shape [bs, 3, 32, 32].
        x_nn: A batch of neighbor images, shape [bs, num_nn, 3, 32, 32].
        device: Device for computation.
        
    Returns:
        standard_recon: Standard reconstructions from model(x).
        neighborhood_recon_image: For visualization, one neighborhood reconstruction per sample
                                  (using the first neighbor's approximation).
    """
    model.eval()
    with torch.no_grad():
        # Standard reconstruction
        standard_recon = model(x.to(device))
        
        # Encode central images
        z_c = model.encoder(x.to(device))
        bs, num_nn, C, H, W = x_nn.size()
        z_dim = z_c.size(1)
        # Encode neighbor images: reshape from [bs, num_nn, 3, 32, 32] -> [bs*num_nn, 3, 32, 32]
        z_nn = model.encoder(x_nn.view(bs * num_nn, C, H, W)).view(bs, num_nn, z_dim)
        # Compute neighborhood reconstruction using the approximation
        n_recon = model.neighborhood_recon(z_c, z_nn)
        # For visualization, select the approximation corresponding to the first neighbor
        neighborhood_recon_image = n_recon[:, 0, :].view(bs, C, H, W)
    return standard_recon, neighborhood_recon_image

##############################################
# New helper function for synthetic visualization (manifold comparison)
##############################################
def visualize_manifold_comparison(noisy_data, vanilla_recon, nrae_recon, mode_label="NRAE-Q", save_path=None):
    """
    Creates a side-by-side plot comparing:
      - Noisy training data,
      - The reconstructed manifold from a vanilla autoencoder, and
      - The smooth manifold produced by the NRAE variant.
    Overlays error vectors (arrows) from the noisy data points to their corresponding reconstructions.
    
    Args:
        noisy_data: NumPy array of shape (N, 2) representing the noisy training points.
        vanilla_recon: NumPy array of shape (N, 2) from the vanilla AE reconstruction.
        nrae_recon: NumPy array of shape (N, 2) from the NRAE reconstruction.
        mode_label: String label for the NRAE mode ("NRAE-Q" or "NRAE-L").
        save_path: (optional) Path to save the figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Noisy Training Data
    axs[0].scatter(noisy_data[:, 0], noisy_data[:, 1], color='blue', label='Noisy Data')
    axs[0].set_title("Noisy Training Data")
    axs[0].legend()
    
    # Panel 2: Vanilla AE Reconstruction with Error Vectors
    axs[1].scatter(vanilla_recon[:, 0], vanilla_recon[:, 1], color='red', label='Vanilla AE')
    for i in range(noisy_data.shape[0]):
        axs[1].arrow(noisy_data[i, 0], noisy_data[i, 1],
                     vanilla_recon[i, 0] - noisy_data[i, 0],
                     vanilla_recon[i, 1] - noisy_data[i, 1],
                     head_width=0.03, head_length=0.05, fc='gray', ec='gray', alpha=0.6)
    axs[1].set_title("Vanilla AE Reconstruction")
    axs[1].legend()
    
    # Panel 3: NRAE Reconstruction with Error Vectors
    axs[2].scatter(nrae_recon[:, 0], nrae_recon[:, 1], color='green', label=mode_label)
    for i in range(noisy_data.shape[0]):
        axs[2].arrow(noisy_data[i, 0], noisy_data[i, 1],
                     nrae_recon[i, 0] - noisy_data[i, 0],
                     nrae_recon[i, 1] - noisy_data[i, 1],
                     head_width=0.03, head_length=0.05, fc='gray', ec='gray', alpha=0.6)
    axs[2].set_title(f"{mode_label} Reconstruction")
    axs[2].legend()
    
    plt.suptitle("Comparison of Noisy Data, Vanilla AE, and NRAE Reconstructions", fontsize=16)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Manifold comparison figure saved to {save_path}")
    plt.show()


##############################################
# Main run function
##############################################
def run(cfg):
    # -----------------------------
    # Setup seeds
    # -----------------------------
    seed = cfg.get("seed", 1)
    print(f"Running with random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------------
    # Setup device
    # -----------------------------
    device = cfg.device

    # -----------------------------
    # Setup Dataloaders
    # -----------------------------
    d_datasets = {}
    d_dataloaders = {}
    for key, dataloader_cfg in cfg["data"].items():
        d_datasets[key] = get_dataset(dataloader_cfg)
        d_dataloaders[key] = get_dataloader(dataloader_cfg)

    # -----------------------------
    # Setup Model
    # -----------------------------
    model = get_model(cfg['model']).to(device)
    model_name = cfg['model']['arch'].upper()
    # If using neighborhood graphs, set the corresponding attribute.
    if cfg["data"]["training"].get("graph", False):
        model.dist_indices = d_datasets['training'].dist_mat_indices

    # -----------------------------
    # Setup optimizer & scheduler
    # -----------------------------
    params = {k: v for k, v in cfg['optimizer'].items() if k != "name"}
    optimizer = torch.optim.Adam(model.parameters(), **params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # -----------------------------
    # Training Loop
    # -----------------------------
    all_train_losses = []
    all_test_losses = []
    num_epochs = cfg['training']['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        training_loss = []

        # Use graph branch only if explicitly enabled in the training config.
        if cfg["data"]["training"].get("graph", False):
            for x, x_nn in d_dataloaders['training']:
                x = x.to(device)
                x_nn = x_nn.to(device)
                train_dict = model.train_step(x, x_nn, optimizer)
                training_loss.append(train_dict["loss"])
        else:
            for x in d_dataloaders['training']:
                x = x.to(device)
                train_dict = model.train_step(x, optimizer)
                training_loss.append(train_dict["loss"])

        avg_train_loss = sum(training_loss) / len(training_loss)
        all_train_losses.append(avg_train_loss)
        print(f"Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}")

        # -----------------------------
        # Compute test loss
        # -----------------------------
        model.eval()
        test_loss = []
        with torch.no_grad():
            for x in d_dataloaders['test']:
                x = x.to(device)
                if hasattr(model, "test_step"):
                    out_dict = model.test_step(x)
                    loss = out_dict["loss"]
                else:
                    out = model(x)
                    if isinstance(out, dict):
                        loss = out.get("loss", 0.0)
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

    # -----------------------------
    # Visualization: CIFAR10 reconstruction GIF (if available)
    # -----------------------------
    if hasattr(model, "cifar10_visualize"):
        gif_path = os.path.join(cfg['logdir'], f'{type(model).__name__}.gif')
        model.cifar10_visualize(d_datasets['training'].data, device, gif_path)
    
    # -----------------------------
    # Plot and save loss curves
    # -----------------------------
    epochs_arr = np.arange(1, num_epochs + 1)
    plot_loss_curves(epochs_arr, all_train_losses, all_test_losses, cfg['logdir'], model_name)

    # -----------------------------
    # Side-by-Side Comparison (Training Set)
    # -----------------------------
    some_batch = next(iter(d_dataloaders['training']))
    if isinstance(some_batch, (list, tuple)):
        x = some_batch[0]  # ignore neighbors if present
    else:
        x = some_batch
    x = x.to(device)
    with torch.no_grad():
        recon = model(x)
        if isinstance(recon, dict) and 'recon' in recon:
            recon = recon['recon']
    x_cpu = x.detach().cpu()[:16]
    recon_cpu = recon.detach().cpu()[:16]
    comparison_path = os.path.join(cfg['logdir'], f"{model_name}_train_comparison.png")
    compare_original_and_reconstructions(x_cpu, recon_cpu, save_path=comparison_path, nrow=8)

    # -----------------------------
    # New Visualization: Compare Standard vs. Neighborhood Reconstructions
    # -----------------------------
    # This section only makes sense if the model is running in NRAE mode.
    if hasattr(model, "neighborhood_recon") and getattr(model, "nrae_mode", False):
        some_batch = next(iter(d_dataloaders['training']))
        if isinstance(some_batch, (list, tuple)) and len(some_batch) >= 2:
            x = some_batch[0].to(device)
            x_nn = some_batch[1].to(device)
            std_recon, nbh_recon = compare_standard_and_neighborhood_recon(model, x, x_nn, device)
            grid_std = make_grid(std_recon, nrow=8, normalize=True, pad_value=1.0)
            grid_nbh = make_grid(nbh_recon, nrow=8, normalize=True, pad_value=1.0)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(grid_std.permute(1, 2, 0).cpu().numpy())
            plt.title("Standard Reconstructions")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(grid_nbh.permute(1, 2, 0).cpu().numpy())
            plt.title("Neighborhood Reconstructions")
            plt.axis("off")
            plt.show()
        else:
            print("Neighborhood batch not available for additional visualization.")
    
    # -----------------------------
    # New Synthetic Visualization: Compare Noisy Data, Vanilla AE, and NRAE-Q Reconstructions
    # -----------------------------
    # This visualization is for a synthetic 1D manifold experiment (2D points).
    # Replace the dummy data below with your actual synthetic experiment data.
    synthetic_save_path = os.path.join(cfg['logdir'], "manifold_comparison.png")
    # Replace the dummy data below with your actual synthetic experiment data.
    N = 100
    x_vals = np.linspace(-3, 3, N)
    noisy_data = np.vstack([x_vals, np.sin(x_vals) + np.random.normal(scale=0.2, size=N)]).T
    vanilla_recon = np.vstack([x_vals, np.sin(x_vals) + np.random.normal(scale=0.5, size=N)]).T

    # If running NRAE-L, pass "NRAE-L"; for NRAE-Q, pass "NRAE-Q"
    nrae_recon = np.vstack([x_vals, np.sin(x_vals)]).T
    visualize_manifold_comparison(noisy_data, vanilla_recon, nrae_recon, mode_label="NRAE-L", save_path=synthetic_save_path)

    # -----------------------------
    # Save Rotated Sequence as a Single Image
    # -----------------------------
    sample_img_data = d_datasets['training'].data[0]
    if torch.is_tensor(sample_img_data):
        sample_img_data = sample_img_data.detach().cpu().numpy()
    if sample_img_data.ndim == 2:
        sample_img_data = (sample_img_data * 255).astype(np.uint8)
    elif sample_img_data.ndim == 3 and sample_img_data.shape[0] in [1, 3]:
        sample_img_data = np.transpose(sample_img_data, (1, 2, 0))
        sample_img_data = (sample_img_data * 255).astype(np.uint8)
    sample_img = Image.fromarray(sample_img_data)
    sample_img = sample_img.resize((112, 112))
    rotated_sequence_path = os.path.join(cfg['logdir'], f"{model_name}_rotated_sample.png")
    save_rotated_sequence(sample_img, rotated_sequence_path, num_frames=20)

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
        assert "=" not in key, "Optional arguments should be separated by space."
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
    print(f"Config saved as {copied_yml}")
    run(cfg)
