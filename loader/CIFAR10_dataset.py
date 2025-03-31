import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

class RotatedShiftedCIFAR10(CIFAR10):
    """
    Select a single class in CIFAR10, then make a rotated or shifted dataset
    from that single image. Similar to RotatedShiftedFashionMNIST, but for CIFAR10.
    """
    def __init__(self, root, type='rotate', digit=0, graph=False, download=True, **kwargs):
        """
        Args:
            root (str): Root directory of dataset where the data will be stored.
            type (str): 'rotate' or 'shift'. Determines which transform to apply.
            digit (int): Class label to pick from [0..9]. This picks the first example 
                         whose label == digit.
            graph (bool): Whether to build the neighborhood graph (for methods like NRAE).
            download (bool): If True, downloads the dataset from the internet.
            **kwargs: Additional keyword arguments for transformation settings, for example:
                - num_rotate (int): Number of rotations if type='rotate'.
                - rotate_range (float): Maximum rotation angle in degrees if type='rotate'.
                - shift_range (int): Maximum horizontal shift if type='shift'.
                - graph['num_nn'] (int): Number of nearest neighbors for the graph.
                - graph['bs_nn'] (int): Number of neighbors to sample at each __getitem__.
                - etc.
        """
        assert type in ['rotate', 'shift'], "type must be 'rotate' or 'shift'."
        super(RotatedShiftedCIFAR10, self).__init__(root=root, download=download)
        
        # Convert the CIFAR10 numpy array to a torch tensor and permute dimensions to [N, C, H, W]
        data = torch.tensor(self.data)  # shape: [N, 32, 32, 3]
        data = data.permute(0, 3, 1, 2)  # shape: [N, 3, 32, 32]
        data = data.to(torch.float32) / 255.0
        
        targets = self.targets  # list of labels
        
        # Find the first occurrence of the requested class (digit)
        idx_to_use = None
        for idx, label in enumerate(targets):
            if label == digit:
                idx_to_use = idx
                break
        if idx_to_use is None:
            raise ValueError(f"No samples found in CIFAR10 with label == {digit}.")
        
        # Keep just that single image
        single_image = data[idx_to_use]  # shape: [3, 32, 32]
        
        # Generate the dataset by applying the specified transformation
        self.kwargs = kwargs
        self.data = self.get_data(type, single_image)
        
        print(f"CIFAR10 dataset ready. {self.data.size()} (type={type}, digit={digit})")
        
        # If graph=True, build the nearest neighbor graph for neighborhood-based methods
        self.graph = graph
        if self.graph:
            self.set_graph()

    def get_data(self, type, single_image):
        """
        Generate a stack of rotated or shifted images from the single_image.
        Returns a tensor of shape [M, 3, 32, 32], where M is the number of transforms.
        """
        data_list = []
        if type == 'rotate':
            num_rotate = self.kwargs.get('num_rotate', 100)
            rotate_range = self.kwargs.get('rotate_range', 180)
            # Rotate from 0 up to rotate_range in evenly spaced steps
            for i in range(num_rotate):
                angle = rotate_range * i / num_rotate
                transformed_data = transforms.functional.affine(
                    img=single_image, 
                    angle=angle, 
                    translate=[0, 0], 
                    scale=1.0, 
                    shear=0
                )
                data_list.append(transformed_data)
        elif type == 'shift':
            shift_range = self.kwargs.get('shift_range', 10)
            # Shift horizontally from -shift_range to +shift_range
            shift_min = -abs(shift_range)
            shift_max = abs(shift_range)
            for sh in range(shift_min, shift_max):
                transformed_data = transforms.functional.affine(
                    img=single_image, 
                    angle=0, 
                    translate=[sh, 0], 
                    scale=0.7,  # optional scale adjustment
                    shear=0
                )
                data_list.append(transformed_data)
        # Stack into a single tensor [M, 3, 32, 32]
        data_list = torch.stack(data_list, dim=0)
        return data_list

    def visualize_data(self, training_data, test_data):
        f = plt.figure()
        plt.scatter(training_data[:, 0], training_data[:, 1], s=50, label='training data')
        plt.plot(test_data[:, 0], test_data[:, 1], linewidth=5, c='k', label='data manifold')
        plt.title('Training Data and Ground Truth Data Manifold')
        plt.legend(loc='upper left')
        plt.xlim(-4, 4)
        plt.ylim(-1.5, 1.5)
        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()
        return f_arr

    def visualize_graph(self, data, _, dist_mat_indices, model=None, device="cpu"):
        """
        Visualize latent space with neighborhood graph connections.
        Assumes a provided model with a 2D encoder output.
        """
        if model is None:
            raise ValueError("Model must be provided to visualize the graph.")
        
        model.eval()
        with torch.no_grad():
            encoded = model.encoder(data.to(device)).cpu()  # shape: [N, 2]
        encoded = encoded.numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot points
        ax.scatter(encoded[:, 0], encoded[:, 1], s=30, c='blue', alpha=0.6, label="Latent points")
        
        # Draw lines between each point and its neighbors
        for i in range(len(encoded)):
            neighbors = dist_mat_indices[i]
            for j in neighbors:
                x_vals = [encoded[i, 0], encoded[j, 0]]
                y_vals = [encoded[i, 1], encoded[j, 1]]
                ax.plot(x_vals, y_vals, c='gray', alpha=0.3)
        
        ax.set_title("Neighborhood Graph in Latent Space")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.legend()
        ax.grid(True)
        fig.canvas.draw()
        fig_arr = np.array(fig.canvas.renderer._renderer)
        plt.close(fig)
        return fig_arr

    def set_graph(self):
        """
        Builds a nearest-neighbor graph over self.data using torch.cdist.
        """
        # Flatten each image to [M, 3*32*32]
        data_temp = self.data.view(len(self.data), -1).clone()
        dist_mat = torch.cdist(data_temp, data_temp)  # [M, M]
        num_nn = self.graph.get('num_nn', 5)
        dist_mat_indices = torch.topk(
            dist_mat, 
            k=num_nn + 1, 
            dim=1, 
            largest=False, 
            sorted=True
        )
        # Skip the first index (self) and keep the rest
        self.dist_mat_indices = dist_mat_indices.indices[:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        If graph=True, returns (center, neighbors) for neighborhood-based training;
        otherwise returns the single transformed image.
        """
        if self.graph:
            bs_nn = self.graph['bs_nn']  # number of neighbors to sample
            include_center = self.graph.get('include_center', True)
            replace = self.graph.get('replace', False)
            x_c = self.data[idx]
            if include_center:
                nn_indices = np.random.choice(range(self.graph['num_nn']), bs_nn - 1, replace=replace)
                x_nn = self.data[self.dist_mat_indices[idx, nn_indices]]
                return x_c, torch.cat([x_c.unsqueeze(0), x_nn], dim=0)
            else:
                nn_indices = np.random.choice(range(self.graph['num_nn']), bs_nn, replace=replace)
                x_nn = self.data[self.dist_mat_indices[idx, nn_indices]]
                return x_c, x_nn
        else:
            return self.data[idx]
