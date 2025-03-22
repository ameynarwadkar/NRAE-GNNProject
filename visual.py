import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Assume `latent_space` is the encoded representation of the dataset
latent_space = model.encoder(data)  # `model.encoder(data)` is the encoder part of your NRAE model

# For PCA (if you want to use PCA for dimensionality reduction)
pca = PCA(n_components=2)
latent_space_2d = pca.fit_transform(latent_space)

# For t-SNE (for better visualization of high-dimensional structures)
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
latent_space_2d_tsne = tsne.fit_transform(latent_space)

# Plot PCA Results
plt.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar()  # Colorbar to show labels/classes
plt.title('2D PCA of Latent Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Plot t-SNE Results
plt.scatter(latent_space_2d_tsne[:, 0], latent_space_2d_tsne[:, 1], c=labels, cmap='viridis')
plt.colorbar()  # Colorbar to show labels/classes
plt.title('2D t-SNE of Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
