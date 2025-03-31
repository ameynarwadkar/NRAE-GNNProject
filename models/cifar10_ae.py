import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import imageio

# Import the kernel function from your original AE implementation.
# If you prefer, you can also re-implement it here.
from models.ae import get_kernel_function

########################################
# Encoder and Decoder for CIFAR10 Images
########################################
class CIFAR10Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Encoder, self).__init__()
        # Convolutional encoder for CIFAR10 (input: [B, 3, 32, 32])
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)   # -> [B, 32, 16, 16]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> [B, 64, 8, 8]
        self.fc = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten to [B, 64*8*8]
        z = self.fc(x)
        return z

class CIFAR10Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Decoder, self).__init__()
        # Map latent vector back to feature map
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        # Transposed convolutions to upsample back to [B, 3, 32, 32]
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> [B, 32, 16, 16]
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)   # -> [B, 3, 32, 32]

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 64, 8, 8)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # output in [0,1]
        return x

########################################
# CIFAR10AE with Optional NRAE Functionality
########################################
class CIFAR10AE(nn.Module):
    def __init__(self, latent_dim=128, nrae_mode=False, approx_order=1, kernel=None):
        """
        Args:
            latent_dim (int): Dimensionality of the latent space.
            nrae_mode (bool): If True, the model computes neighborhood reconstruction loss.
            approx_order (int): Order of local approximation (1 for linear, 2 for quadratic).
            kernel (dict): Dictionary defining the kernel for neighborhood weighting.
        """
        super(CIFAR10AE, self).__init__()
        self.encoder = CIFAR10Encoder(latent_dim)
        self.decoder = CIFAR10Decoder(latent_dim)
        self.nrae_mode = nrae_mode
        if self.nrae_mode:
            self.approx_order = approx_order
            # If kernel is provided, get the kernel function; otherwise default weight =1
            self.kernel_func = get_kernel_function(kernel) if kernel is not None else None

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    ########################################
    # Standard Training Step
    ########################################
    def standard_train_step(self, x, optimizer):
        optimizer.zero_grad()
        recon = self(x)
        loss = ((recon - x) ** 2).view(x.size(0), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    ########################################
    # Neighborhood Reconstruction Functions
    ########################################
    def jacobian(self, z, dz, create_graph=True):
        bs, num_nn, z_dim = dz.size()
        v = dz.view(-1, z_dim)  # shape: [bs*num_nn, z_dim]
        inputs = z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim)
        # Compute Jacobian-vector product using autograd.functional.jvp
        _, jvp = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)
        return jvp.view(bs, num_nn, -1)

    def jacobian_and_hessian(self, z, dz, create_graph=True):
        bs, num_nn, z_dim = dz.size()
        v = dz.view(-1, z_dim)
        inputs = z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim)
        def jac_temp(inp):
            jvp = torch.autograd.functional.jvp(self.decoder, inp, v=v, create_graph=create_graph)[1]
            return jvp.view(bs, num_nn, -1)
        jac, hess = torch.autograd.functional.jvp(jac_temp, inputs, v=v, create_graph=create_graph)
        return jac.view(bs, num_nn, -1), hess.view(bs, num_nn, -1)

    def neighborhood_recon(self, z_c, z_nn):
        """
        Computes the local approximation of the decoder output.
        Args:
            z_c: Encoded central images [bs, z_dim]
            z_nn: Encoded neighbor images [bs, num_nn, z_dim]
        Returns:
            n_recon: Approximated reconstructions for neighbors, shape [bs, num_nn, x_dim]
        """
        recon = self.decoder(z_c)  # [bs, x_dim]
        recon_x = recon.view(z_c.size(0), 1, -1)  # [bs, 1, x_dim]
        dz = z_nn - z_c.unsqueeze(1)             # [bs, num_nn, z_dim]
        if self.approx_order == 1:
            Jdz = self.jacobian(z_c, dz)           # [bs, num_nn, x_dim]
            n_recon = recon_x + Jdz
        elif self.approx_order == 2:
            Jdz, dzHdz = self.jacobian_and_hessian(z_c, dz)
            n_recon = recon_x + Jdz + 0.5 * dzHdz
        else:
            raise ValueError("Unsupported approx_order: choose 1 or 2.")
        return n_recon

    def nrae_train_step(self, x, x_nn, optimizer):
        """
        Neighborhood reconstruction training step.
        Args:
            x: Central images, shape [bs, 3, 32, 32]
            x_nn: Neighbor images, shape [bs, num_nn, 3, 32, 32]
            optimizer: Optimizer to update the network.
        """
        optimizer.zero_grad()
        bs = x_nn.size(0)
        num_nn = x_nn.size(1)
        z_c = self.encoder(x)  # [bs, z_dim]
        z_dim = z_c.size(1)
        # Encode neighbors: reshape from [bs, num_nn, 3, 32, 32] -> [bs*num_nn, 3, 32, 32]
        z_nn = self.encoder(x_nn.view(bs * num_nn, *x_nn.shape[2:])).view(bs, num_nn, z_dim)
        n_recon = self.neighborhood_recon(z_c, z_nn)  # [bs, num_nn, x_dim]
        # Compute neighborhood loss (MSE)
        n_loss = torch.norm(x_nn.view(bs, num_nn, -1) - n_recon, dim=2) ** 2
        if self.kernel_func is not None:
            weights = self.kernel_func(x, x_nn)  # expects shape [bs, num_nn]
        else:
            weights = 1.0
        loss = (weights * n_loss).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    ########################################
    # Unified Train Step
    ########################################
    def train_step(self, *args, **kwargs):
        """
        If in NRAE mode, expects (x, x_nn, optimizer);
        else expects (x, optimizer).
        """
        if self.nrae_mode:
            if len(args) >= 3:
                x, x_nn, optimizer = args[0], args[1], args[2]
                return self.nrae_train_step(x, x_nn, optimizer)
            else:
                raise ValueError("For NRAE mode, expected arguments: x, x_nn, optimizer.")
        else:
            if len(args) >= 2:
                x, optimizer = args[0], args[1]
                return self.standard_train_step(x, optimizer)
            else:
                raise ValueError("Expected arguments: x, optimizer for standard mode.")

    ########################################
    # CIFAR10 Visualization Helpers
    ########################################
    def cifar10_visualize(self, data, device, sname, n_vis_step=16):
        """
        Visualizes a subset of CIFAR10 reconstructions and saves a grid image.
        """
        # Select n_vis_step images evenly from the dataset.
        vis_data = []
        step = max(1, len(data) // n_vis_step)
        for i in range(0, len(data), step):
            if len(vis_data) >= n_vis_step:
                break
            vis_data.append(data[i])
        vis_data = torch.stack(vis_data, dim=0).to(device)
        
        with torch.no_grad():
            recon = self(vis_data)
        recon = recon.detach().cpu()

        grid = make_grid(recon, nrow=n_vis_step, normalize=True, pad_value=1.0)
        os.makedirs(os.path.dirname(sname), exist_ok=True)
        save_image(grid, sname)
        print(f"Visualization saved as {sname}")

        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Reconstructed Images")
        plt.show()

    def create_latent_sweep_gif(self, device, sname, num_frames=20):
        """
        Generates a GIF by sweeping through the latent space.
        (Assumes a 1D latent space for simplicity.)
        """
        self.eval()
        with torch.no_grad():
            z_min, z_max = -3.0, 3.0
            latent_sweep = torch.linspace(z_min, z_max, num_frames).to(device).unsqueeze(1)
            gen_imgs = self.decoder(latent_sweep).detach().cpu()
        frames = []
        for i in range(gen_imgs.size(0)):
            grid = make_grid(gen_imgs[i].unsqueeze(0), normalize=True, pad_value=1.0)
            frame = grid.permute(1, 2, 0).numpy()
            frames.append(frame)
        imageio.mimsave(sname, frames, duration=0.5)
        print(f"Latent sweep GIF saved as {sname}")

    def compare_standard_and_neighborhood_recon(model, x, x_nn, device):
        """
        Compute and return both standard and neighborhood reconstructions for a batch.
        
        Args:
            model: Your CIFAR10AE model.
            x: A batch of central images, shape [bs, 3, 32, 32].
            x_nn: A batch of neighbor images, shape [bs, num_nn, 3, 32, 32].
            device: Device for computation.
            
        Returns:
            standard_recon: The standard reconstruction from model(x).
            neighborhood_recon_image: For visualization, we select one neighborhood reconstruction 
                                    per sample (e.g., the first neighbor's approximation), reshaped to [bs, 3, 32, 32].
        """
        model.eval()
        with torch.no_grad():
            # Standard reconstruction
            standard_recon = model(x.to(device))
            
            # Encode central images
            z_c = model.encoder(x.to(device))
            bs, num_nn, C, H, W = x_nn.size()
            z_dim = z_c.size(1)
            # Encode neighbor images: reshape to [bs*num_nn, 3, 32, 32] then back to [bs, num_nn, z_dim]
            z_nn = model.encoder(x_nn.view(bs * num_nn, C, H, W)).view(bs, num_nn, z_dim)
            # Compute neighborhood reconstruction using the approximation
            n_recon = model.neighborhood_recon(z_c, z_nn)
            # For visualization, pick the reconstruction from the first neighbor as a representative.
            neighborhood_recon_image = n_recon[:, 0, :].view(bs, C, H, W)
        return standard_recon, neighborhood_recon_image

    def visualize_manifold_comparison(noisy_data, vanilla_recon, nrae_recon,mode_label="NRAE-Q" ,save_path=None):
        """
        Creates a side-by-side plot comparing:
        - Noisy training data,
        - The reconstructed manifold from a vanilla autoencoder, and
        - The smooth manifold produced by NRAE-Q.
        Overlays error vectors (arrows) from the noisy data points to their respective reconstructions.
        
        Args:
            noisy_data: NumPy array of shape (N, 2) representing the noisy training points.
            vanilla_recon: NumPy array of shape (N, 2) from the vanilla AE reconstruction.
            nrae_recon: NumPy array of shape (N, 2) from the NRAE-Q reconstruction.
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
        
        # Panel 3: NRAE-Q Reconstruction with Error Vectors
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