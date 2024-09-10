from sklearn.mixture import GaussianMixture
from ZarrDataLoader import ZarrDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from sklearn.metrics import silhouette_score
import torch
from tqdm import tqdm
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def model_prediction(model, dataloader, saved_weights, device, output_save_path):
    '''
    Function for using a model in prediction mode to generate and save latent space representations.

    Parameters
    ----------
    model : PyTorch model instance
        Model with trained parameters.

    dataloader : PyTorch DataLoader instance
        Loads data from disk into memory.

    saved_weights : str
        Path to the model's saved weights.

    device : str
        Device to run the model on ('cuda' or 'cpu').

    output_save_path : str
        Path to save the output latent space representations.
    '''
    print('Evaluating data using the model...')
    device = torch.device(device)
    model.load_state_dict(torch.load(saved_weights, map_location=device))
    model = model.double()  # Convert entire model to double precision
    model.eval()
    model.to(device)

    z_array = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing", unit="batch"):
            batch_size, mini_batch, channels, height, width = batch.size()
            inputs = batch.view(batch_size * mini_batch, channels, height, width)
            inputs = inputs.to(device, dtype=torch.double)

            # Adjust this to match your model's method of obtaining latent representations
            z = model.encoder(inputs)

            z_array.append(z.cpu().numpy())

    z_array = np.concatenate(z_array, axis=0)

    logging.info(f"create dir: {fname}")
    os.makedirs(output_save_path, exist_ok=True)
    os.makedirs(os.path.join(output_save_path, os.pardir, 'GMM'), exist_ok=True)

    np.save(os.path.join(output_save_path, 'Z_AEC.npy'), z_array)
    print(f'Latent space representations saved to {output_save_path}')


def gmm(z_array, n_clusters):
    """
    Initialize clusters using Gaussian Mixture Model algorithm.
    """
    # Initialize with K-Means
    km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, n_init=100, random_state=2009)
    gmm_labels = km.fit_predict(z_array)
    labels = km.labels_
    centroids = km.cluster_centers_
    wcss = km.inertia_
    silhouette_avg = silhouette_score(z_array, gmm_labels)

    # Estimate initial GMM weights
    labels_unique, counts = np.unique(labels, return_counts=True)
    M = z_array.shape[0]
    gmm_weights = counts / M

    # Initialize and fit GMM
    GMM = GaussianMixture(n_components=n_clusters, max_iter=1000, n_init=1,
                          weights_init=gmm_weights, means_init=centroids)
    np.seterr(under='ignore')
    labels = GMM.fit_predict(z_array)
    centroids = GMM.means_
    return labels, centroids, wcss, silhouette_avg


def plot_silhouette_scores(n_clusters_list, silhouette_scores):
    """
    Plot silhouette scores against number of clusters.
    """
    plt.figure()
    plt.plot(n_clusters_list, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette analysis For Optimal k')
    return plt

def plot_elbow_method(n_clusters_list, wcss):
    """
    Plot the elbow method graph.
    """
    plt.figure()
    plt.plot(n_clusters_list, wcss, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('The Elbow Method showing the optimal k')
    return plt


def view_TSNE(z_array, labels, title):
    """
    Visualize the latent space using t-SNE.
    """
    tsne = TSNE(n_components=2, random_state=2009)
    tsne_results = tsne.fit_transform(z_array)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    return plt


def init_clustering(saved_weights, n_clusters_list, fname):
    """
    Initialize clustering with GMM and save cluster labels and centroids.
    """

    silhouette_scores = []
    wcss_list = []

    # Load dataset from saved latent space
    dataset = np.load(os.path.join(fname, 'Z_AEC.npy'))
    logging.info(f'Dataset has {len(dataset)} samples.')

    savepath_exp = os.path.abspath(os.path.join(saved_weights, os.pardir, 'GMM'))
    os.makedirs(savepath_exp, exist_ok=True)

    for n_clusters in n_clusters_list:
        logging.info('-'*100)
        logging.info(f'GMM Run: n_clusters={n_clusters}')
        logging.info('-' * 25)

        labels, centroids, wcss, silhouette_avg = gmm(dataset, n_clusters)
        silhouette_scores.append(silhouette_avg)
        wcss_list.append(wcss)


        # Save labels and centroids
        np.save(os.path.join(savepath_exp, f'labels_{n_clusters}.npy'), labels)
        np.save(os.path.join(savepath_exp, f'centroids_{n_clusters}.npy'), centroids)

        # Visualize using t-SNE
        fig = view_TSNE(dataset, labels, f'GMM with {n_clusters} Clusters')
        fig.savefig(os.path.join(savepath_exp, f't-SNE_{n_clusters}.png'), dpi=300)
        plt.close(fig)

        logging.info(f'GMM clustering with {n_clusters} clusters complete and saved.')
    fig = plot_silhouette_scores(n_clusters_list, silhouette_scores)
    fig.savefig(os.path.join(savepath_exp, f'silhouette.png'), dpi=300)
    plt.close(fig)
    fig = plot_elbow_method(n_clusters_list, wcss)
    fig.savefig(os.path.join(savepath_exp, f'wcss.png'), dpi=300)
    plt.close(fig)

# Example usage
# Prepare your dataset and dataloader
transform_pipeline = transforms.Compose([
    ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
    lambda x: x.double(),
])
#dataset = ZarrDataset('/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_5758.zarr', 4,
#                           transform=transform_pipeline)
#dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5, pin_memory=True)
saved_weights = './preliminary_weights/Best_Model.pt'
fname = os.path.abspath(os.path.join('./preliminary_weights', 'Prediction'))
#model_prediction(UNet(), dataloader, saved_weights, 'cuda', fname)
n_clusters_list = [4, 6, 8, 10, 12, 14]  # Example cluster sizes to try
init_clustering(saved_weights, n_clusters_list, fname)