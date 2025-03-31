# Neighborhood Reconstructing Autoencoders for Manifold Learning – Experimental Study

This repository contains the implementation and experimental evaluation of Neighborhood Reconstructing Autoencoders (NRAE) for manifold learning. Our approach extends traditional autoencoder frameworks by incorporating a neighborhood reconstruction loss that explicitly enforces local geometric consistency, resulting in improved reconstruction fidelity and more robust latent space representations.

## Overview

In this project, we compare three model variants:
- **Standard Autoencoder (AE)**
- **NRAEL**: Neighborhood Reconstructing Autoencoder using a first-order (linear) approximation.
- **NRAEQ**: Neighborhood Reconstructing Autoencoder using a second-order (quadratic) approximation.

Our experiments span several datasets, demonstrating the benefits of neighborhood-based reconstruction:
- **CIFAR10:** Evaluating rotational invariance and denoising performance.
- **FashionMNIST:** Assessing rotational variations and geometric preservation.
- **MNIST:** Analyzing reconstruction fidelity and latent space regularity.
- **Synthetic Datasets:** Including Sine Curve for denoising and Swiss Roll for geometry preservation.

## Experimental Highlights

- **Rotational Invariance:**  
  On CIFAR10 and FashionMNIST, NRAEL and NRAEQ capture rotational transformations far more effectively than standard VAE models by maintaining local geometric continuity.

- **Denoising and Geometry Preservation:**  
  Synthetic experiments on a noisy Sine Curve and the Swiss Roll show that the neighborhood reconstruction loss not only enhances denoising capabilities but also preserves the intrinsic manifold structure. In particular, the second-order NRAEQ model yields smoother latent space representations, as evidenced by scalar curvature analyses.

- **Latent Space Regularity:**  
  Experiments with MNIST, including scalar curvature heatmaps, indicate that increasing the number of nearest neighbors (num_nn) in the reconstruction process leads to a more continuous and coherent latent space.

Refer to the `results/` directory for visual comparisons and detailed figures.

## Repository Structure

- `configs/`: Configuration files for various experiments.
- `models/`: Implementation of autoencoder architectures and neighborhood reconstruction functions.
- `train_*.py`: Training scripts for CIFAR10, FashionMNIST, MNIST, and synthetic datasets.
- `results/`: Visualizations and comparative analyses.
- `GNN_Project.pdf`: The project report detailing methodology, experiments, and conclusions.

## Environment

The project is developed using PyTorch under the following conditions:
- Python 3.8+
- PyTorch 1.8+
- CUDA (if available)
- Additional Python packages: numpy, matplotlib, imageio, argparse, yaml, omegaconf, etc.

## Running the Code

To train a model on a specific dataset, run the corresponding training script. For example:
