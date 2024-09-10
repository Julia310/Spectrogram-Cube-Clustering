#!/usr/bin/env python3

"""William Jenkins
Scripps Institution of Oceanography, UC San Diego
wjenkins [at] ucsd [dot] edu
May 2021

Contains functions, routines, and data recording for DEC model
initialization, training, validation, and inference.
"""

from datetime import datetime
import fnmatch
import os
import pickle
import shutil
import threading
import time

import numpy as np
try:
    import cupy
    from cuml import KMeans, TSNE
    from cuml.metrics.cluster.silhouette_score \
        import cython_silhouette_samples as silhouette_samples
except:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_samples
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
from sklearn.mixture import GaussianMixture
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Cluster import plotting, utils


def batch_eval(dataloader, model, device, mute=True):
    '''Run DEC model in batch_inference mode.

    Parameters
    ----------
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    model : PyTorch model instance
        Model with trained parameters

    device : PyTorch device object ('cpu' or 'gpu')

    mute : Boolean (default=False)
        Verbose mode

    Returns
    -------
    z_array : array (M,D)
        Latent space data (m_samples, d_features)
    '''
    model.eval()
    model.double()

    bsz = dataloader.batch_size
    z_array = np.zeros(
        (len(dataloader.dataset),
        model.encoder.encoder[8].out_features),
        #dtype=np.float32
        dtype=np.float64
    )

    # If the model has the "n_clusters" attribute, then the correct
    # outputs must be collected. In pre-training mode, there is no
    # clustering layer, hence the model only returns 2 quantities; for
    # other modes, the model yields 3 quantities.
    if hasattr(model, 'n_clusters'):
        q_array = np.zeros(
            (len(dataloader.dataset),
            model.n_clusters),
            #dtype=np.float32
            dtype=np.float64
        )
        for b, batch in enumerate(tqdm(dataloader, disable=mute)):
            #print(batch)
            #_, batch = batch
            x = batch.to(device)
            q, _, z = model(x)
            q_array[b * bsz:(b*bsz) + x.size(0), :] = q.detach().cpu().numpy()
            z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()

        labels = np.argmax(q_array.data, axis=1)

        return np.round(q_array, 5), labels, z_array
    else:
        for b, batch in enumerate(tqdm(dataloader, disable=mute)):
            _, batch = batch
            x = batch.to(device)
            _, z = model(x)
            z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
            print(z_array)

        return z_array


def batch_training(model, dataloader, optimizer, metric, device):
    '''Run DEC model in batch_training mode.

    Parameters
    ----------
    model : PyTorch model instance
        Model with untrained parameters.
    
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.
    
    optimizer : PyTorch optimizer instance

    metric : PyTorch metric instance
        Measures the loss function.
    
    device : PyTorch device object ('cpu' or 'gpu')

    Returns
    -------
    model : Pytorch model instance
        Model with trained parameters.
    
    epoch_loss : float
        Loss function value for the given epoch.
    '''
    model.train()

    running_loss = 0.0
    running_size = 0

    pbar = tqdm(
        dataloader,
        leave=True,
        desc="  Training",
        unit="batch",
        postfix={str(metric)[:-2]: "%.6f" % 0.0},
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    for batch in pbar:
        batch_size, mini_batch, channels, height, width = batch.size()
        x = batch.view(batch_size * mini_batch, channels, height, width).to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            #x = x.float()
            model.double()
            x_rec, _ = model(x)
            loss = metric(x_rec, x)
            loss.backward()
            optimizer.step()

        running_loss += loss.cpu().detach().numpy() * x.size(0)
        running_size += x.size(0)

        pbar.set_postfix(
            {str(metric)[:-2]: f"{(running_loss / running_size):.4e}"}
        )

    epoch_loss = running_loss / len(dataloader.dataset)
    return model, epoch_loss


def batch_validation(model, dataloader, metrics, config):
    '''Run DEC model in batch_validation mode.

    Parameters
    ----------
    model : PyTorch model instance
        Model with trained parameters.
    
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    metrics : list
        List of PyTorch metric instances
    
    config : Configuration object
        Object containing information about the experiment configuration

    Returns
    -------
    epoch_loss : float
        Loss function value for the given epoch.
    '''
    if not isinstance(metrics, list):
        metrics = [metrics]

    model.eval()

    running_loss = np.zeros((len(metrics),), dtype=float)
    running_size = np.zeros((len(metrics),), dtype=int)

    pbar = tqdm(
        dataloader,
        leave=True,
        desc="Validation",
        unit="batch",
        postfix={str(metric)[:-2]: "%.6f" % 0.0 for metric in metrics},
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    for batch in pbar:
        batch_size, mini_batch, channels, height, width = batch.size()
        x = batch.view(batch_size * mini_batch, channels, height, width).to(config.device)
        loss = torch.zeros((len(metrics),))
        with torch.no_grad():
            if config.model == 'AEC':
                x_rec, _ = model(x)
            elif config.model == 'DEC':
                _, x_rec, _ = model(x)
            for i, metric in enumerate(metrics):
                loss[i] = metric(x_rec, x)

        running_loss += loss.cpu().detach().numpy() * x.size(0)
        running_size += x.size(0)

        pbar.set_postfix(
            {
                str(metric)[:-2]: f"{(running_loss[i] / running_size[i]):.4e}" \
                    for i, metric in enumerate(metrics)
            }
        )

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


def cluster_metrics(path, labels, x, z, save=True):
    '''Calculates various metrics for clustering performance analysis.
    
    Parameters
    ----------
    path : str
        Path to which clustering metrics results are saved.

    labels : array (M,)
        Sample-wise cluster assignment

    x : array (M, D_)
        Data space, i.e., spectrogram data (M samples, D_ features)

    z : array (M,D)
        Latent space data (M samples, D features)

    save : bool (Default: True)
        Save results to file or not.

    Returns
    -------
    M : array (K,)
        Number of data samples assigned to each class (K labels)

    X_ip_avg : array (K,)
        Inner product between the data space points and their mean.

    X_MSE : array (K, D_)
        Mean squared error between the data space points and their mean.
        (K labels, D_ features)

    X_MSE_avg : array (K,)
        Class-averaged mean squared error between the data space points
        and their mean.

    X_MAE : array (K, D_)
        Mean absolute error between the data space points and their
        mean (K labels, D_ features).

    X_MAE_avg : array (K,)
        Class-averaged mean absolute error between the data space points
        and their mean.

    silh_scores_Z : array (K,)
        Class-averaged silhouette scores of the latent space.

    silh_scores_X : array (K,)
        Class-averaged silhouette scores of the data space.

    df : Pandas DataFrame
        Dataframe containing all metrics results
    '''
    label_list = np.unique(labels)
    n_clusters = len(label_list)

    if torch.cuda.is_available():
        silh_scores_Z = silhouette_samples(z, labels, chunksize=20000)
        silh_scores_Z = cupy.asnumpy(silh_scores_Z)
    else:
        silh_scores_Z = silhouette_samples(z, labels)

    silh_scores_X, _ = silhouette_samples_X(x, labels, RF=3)

    silh_scores_avg_Z = np.mean(silh_scores_Z)
    silh_scores_avg_X = np.mean(silh_scores_X)

    _, _, n, o = x.shape
    M = np.zeros((n_clusters,), dtype=int)
    X_ip_avg = np.zeros((n_clusters,))
    X_MSE = np.zeros((n_clusters, n*o))
    X_MAE = np.zeros((n_clusters, n*o))
    X_MSE_avg = np.zeros((n_clusters,))
    X_MAE_avg = np.zeros((n_clusters,))
    class_silh_scores_Z = np.zeros((n_clusters,))
    class_silh_scores_X = np.zeros((n_clusters,))

    for j in range(n_clusters):

        # Data Space Metrics:
        x_j = np.reshape(x[labels==j], (-1, 8700))
        M[j] = len(x_j)
        x_mean = np.mean(x_j, axis=0).reshape((1,-1))
        x_mean = np.matlib.repmat(x_mean, M[j], 1)
        # Inner Product
        X_ip = linear_kernel(x_j, x_mean[0].reshape(1, -1)).flatten()
        X_ip_avg[j] = np.mean(X_ip)
        # MSE
        X_MSE[j] = mean_squared_error(x_mean, x_j, multioutput='raw_values')
        X_MSE_avg[j] = np.mean(X_MSE)
        # MAE
        X_MAE[j] = mean_absolute_error(x_mean, x_j, multioutput='raw_values')
        X_MAE_avg[j] = np.mean(X_MAE)

        # Silhouette Score - Latent Space
        class_silh_scores_Z[j] = np.mean(silh_scores_Z[labels==j])

        # Silhouette Score - Data Space
        class_silh_scores_X[j] = np.mean(silh_scores_X[labels==j])

    if save:
        np.save(os.path.join(path, 'X_ip'), X_ip_avg)
        np.save(os.path.join(path, 'X_MSE'), X_MSE)
        np.save(os.path.join(path, 'X_MSE_avg'), X_MSE_avg)
        np.save(os.path.join(path, 'X_MAE'), X_MAE)
        np.save(os.path.join(path, 'X_MAE_avg'), X_MAE_avg)
        np.save(os.path.join(path, 'silh_scores_Z'), silh_scores_Z)
        np.save(os.path.join(path, 'silh_scores_X'), silh_scores_X)
        df = pd.DataFrame(
            data={
                'class': label_list,
                'N': M,
                'inner_product': X_ip_avg,
                'MSE_avg': X_MSE_avg,
                'MAE_avg': X_MAE_avg,
                'silh_score_Z': class_silh_scores_Z,
                'silh_score_X': class_silh_scores_X
            }
        )
        df.loc['mean'] = df.mean()
        df.loc['mean']['class', 'N'] = None
        df.loc['mean']['silh_score_Z'] = silh_scores_avg_Z
        df.loc['mean']['silh_score_X'] = silh_scores_avg_X
        df.to_csv(os.path.join(path, 'cluster_performance.csv'))

    return M, X_ip_avg, X_MSE, X_MSE_avg, X_MAE, X_MAE_avg, silh_scores_Z, silh_scores_X, df


def gmm(z_array, n_clusters):
    '''Initialize clusters using Gaussian mixtures model algorithm.

    Parameters
    ----------
    z_array : array (M,D)
        Latent space data (m_samples, d_features)

    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : array (M,)
        Sample-wise cluster assignment

    centroids : array (n_clusters, n_features)
        Cluster centroids
    '''
    M = z_array.shape[0]
    # Initialize w/ K-Means
    km = KMeans(
        n_clusters=n_clusters,
        max_iter=1000,
        n_init=100,
        random_state=2009
    )
    km.fit_predict(z_array)
    labels = km.labels_
    centroids = km.cluster_centers_

    labels, counts = np.unique(labels, return_counts=True)

    # Perform EM
    gmm_weights = np.empty(len(labels))
    for i in range(len(labels)):
        gmm_weights[i] = counts[i] / M

    GMM = GaussianMixture(
        n_components=n_clusters,
        max_iter=1000,
        n_init=1,
        weights_init=gmm_weights,
        means_init=centroids
    )
    np.seterr(under='ignore')
    labels = GMM.fit_predict(z_array)
    centroids = GMM.means_
    return labels, centroids


def gmm_fit(config, z_array, n_clusters):
    '''Perform GMM clustering and save results.

    Parameters
    ----------
    config : Configuration object
        Object containing information about the experiment configuration
    
    z_array : array (M,D)
        Latent space data (m_samples, d_features)

    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : array (M,)
        Sample-wise cluster assignment

    centroids : array (n_clusters, n_features)
        Cluster centroids
    '''
    tic = datetime.now()
    print('Performing GMM...', end="", flush=True)
    labels, centroids = gmm(z_array, n_clusters)
    print('complete.')


    print('Saving data......', end="", flush=True)
    M = z_array.shape[0]
    A = [{'idx': i, 'label': labels[i]} for i in np.arange(M)]
    utils.save_labels(A, config.savepath_run)
    np.save(os.path.join(config.savepath_run, 'labels'), labels)
    np.save(os.path.join(config.savepath_run, 'centroids'), centroids)
    print('complete.')

    #print('Performing clustering metrics...', end='', flush=True)
    #x = np.load(config.fname_dataset + '.npy')
    #_, _, _, _, _, _, silh_scores_Z, silh_scores_X, _ = cluster_metrics(
    #    config.savepath_run,
    #    labels,
    #    x,
    #    z_array
    #)
    #fig1 = plotting.view_silhscore(
    #    silh_scores_Z,
    #    labels,
    #    n_clusters,
    #    config.model,
    #    config.show
    #)
    #fig1.savefig(
    #    os.path.join(config.savepath_run, 'silh_score_Z.png'),
    #    dpi=300,
    #    facecolor='w'
    #)
    #fig2 = plotting.view_silhscore(
    #    silh_scores_X,
    #    labels,
    #    n_clusters,
    #    config.model,
    #    config.show
    #)
    #fig2.savefig(
    #    os.path.join(config.savepath_run, 'silh_score_X.png'),
    #    dpi=300,
    #    facecolor='w'
    #)

    tsne_results = tsne(z_array)
    fig3 = plotting.view_TSNE(tsne_results, labels, 'GMM', config.show)
    fig3.savefig(
        os.path.join(config.savepath_run, 't-SNE.png'),
        dpi=300,
        facecolor='w'
    )
    print('complete.')

    toc = datetime.now()
    print(f'GMM complete at {toc}; time elapsed = {toc-tic}.')


def initialize_clusters(model, dataloader, config, n_clusters=None):
    '''Function selects and performs cluster initialization.
    
    Parameters
    ----------
    model : PyTorch model instance
        Model with trained parameters.
    
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.
    
    config : Configuration object
        Object containing information about the experiment configuration
    
    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : array (M,)
        Sample-wise cluster assignment

    centroids : array (n_clusters, n_features)
        Cluster centroids
    '''
    if config.init == 'load':
        print('Loading cluster initialization...', end='', flush=True)
        path = os.path.abspath(os.path.join(config.saved_weights, os.pardir))
        path = os.path.join(path, 'GMM', f'n_clusters={n_clusters}')

        print(os.path.join(path, 'labels.npy'))
        labels = np.load(os.path.join(path, 'labels.npy'))[config.index_tra]
        centroids = np.load(os.path.join(path, 'centroids.npy'))
    if config.init == "rand": # Random Initialization (for testing)
        print('Initiating clusters with random points...', end='', flush=True)
        labels, centroids = np.random.randint(0, n_clusters, (2300)), np.random.uniform(size=(n_clusters,9))
    else:
        _, _, z_array = batch_eval(dataloader, model, config.device)
        if config.init == "kmeans":
            print('Initiating clusters with k-means...', end="", flush=True)
            labels, centroids = kmeans(z_array, model.n_clusters)
        elif config.init == "gmm": # GMM Initialization:
            print('Initiating clusters with GMM...', end="", flush=True)
            labels, centroids = gmm(z_array, model.n_clusters)

    return labels, centroids


def kmeans(z_array, n_clusters):
    '''Initiate clusters using k-means algorithm.

    Parameters
    ----------
    z_array : array (M,D)
        Latent space data (m_samples, d_features)

    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : array (M,)
        Sample-wise cluster assignment

    centroids : array (n_clusters,)
        Cluster centroids
    '''
    km = KMeans(
        n_clusters=n_clusters,
        max_iter=1000,
        n_init=100,
        random_state=2009
    )
    km.fit_predict(z_array)
    labels = km.labels_
    centroids = km.cluster_centers_
    return labels, centroids


def model_prediction(
        config,
        model,
        dataloader,
        metrics,
    ):
    '''Primary machinery function for using AEC or DEC model in
    prediction (evaluation) mode.
    
    Parameters
    ----------
    config : Configuration object
        Object containing information about the experiment configuration
    
    model : PyTorch model instance
        Model with trained parameters.
    
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.
    
    metrics : list
        List of PyTorch metric instances
    '''
    print(f'Evaluating data using {config.model} model...')
    device = config.device
    n_clusters = config.n_clusters
    savepath = config.savepath_exp

    model.load_state_dict(torch.load(config.saved_weights, map_location=device))
    model.double()  # Convert entire model to double precision
    model.eval()

    bsz = dataloader.batch_size

    z_array = np.zeros((len(dataloader.dataset), model.encoder.encoder[8].out_features), dtype=np.float32)
    xr_array = np.zeros((len(dataloader.dataset), 1, 4, 101), dtype=np.float32)

    pbar = tqdm(
        dataloader,
        leave=True,
        desc="Loading",
        unit="batch",
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    if config.model == 'DEC':
        q_array = np.zeros((len(dataloader.dataset), n_clusters),dtype=np.float32)
        for b, batch in enumerate(pbar):
            #_, batch = batch
            x = batch.to(device)
            q, xr, z = model(x)
            q_array[b * bsz:(b*bsz) + x.size(0), :] = q.detach().cpu().numpy()
            z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
            xr_array[b * bsz:(b*bsz) + x.size(0), :] = xr.detach().cpu().numpy()

        labels = np.argmax(q_array.data, axis=1)
        centroids = model.clustering.weights.detach().cpu().numpy()

        del batch, x, q, xr, z

        time.sleep(1)
        print('Saving data...', end="", flush=True)
        M = q_array.shape[0]
        A = [{'idx': i, 'label': labels[i]} for i in np.arange(M)]
        utils.save_labels(A, savepath)
        np.save(os.path.join(savepath, 'q_DEC'), q_array)
        np.save(os.path.join(savepath, 'Z_DEC'), z_array)
        np.save(os.path.join(savepath, 'Xr_DEC'), xr_array)
        np.save(os.path.join(savepath, 'labels_DEC'), labels)
        np.save(os.path.join(savepath, 'centroids_DEC'), centroids)
        print('complete.')

        print('Performing clustering metrics...', end='', flush=True)
        x = np.load(config.fname_dataset + '.npy')
        _, _, _, _, _, _, silh_scores_Z, silh_scores_X, _ = cluster_metrics(savepath, labels, x, z_array)
        fig = plotting.view_silhscore(silh_scores_Z, labels, n_clusters, config.model, config.show)
        fig.savefig(os.path.join(savepath, 'silh_score_Z.png'), dpi=300, facecolor='w')
        fig = plotting.view_silhscore(silh_scores_X, labels, n_clusters, config.model, config.show)
        fig.savefig(os.path.join(savepath, 'silh_score_X.png'), dpi=300, facecolor='w')
        print('complete.')

        print('Creating figures...')
        AEC_configpath = os.path.abspath(os.path.join(savepath, os.pardir, os.pardir))
        AEC_configname = fnmatch.filter([f for f in os.listdir(AEC_configpath) if os.path.isfile(os.path.join(AEC_configpath, f))], '*.pkl')[0]
        AEC_configpath = pickle.load(open(os.path.join(AEC_configpath, AEC_configname), 'rb'))['saved_weights']

        fignames = [
            'T-SNE',
            'Gallery',
            'LatentSpace',
            'CDF',
            'PDF'
        ]
        figpaths = [os.path.join(savepath, name) for name in fignames]
        [os.makedirs(path, exist_ok=True) for path in figpaths]

        AEC_loadpath = os.path.abspath(os.path.join(AEC_configpath, os.pardir))
        z_array_AEC = np.load(os.path.join(AEC_loadpath, 'Prediction', 'Z_AEC.npy'))
        labels_GMM = np.load(os.path.join(AEC_loadpath, 'GMM', f'n_clusters={n_clusters}', 'labels.npy'))
        centroids_GMM = np.load(os.path.join(AEC_loadpath, 'GMM', f'n_clusters={n_clusters}', 'centroids.npy'))

        tsne_results = tsne(z_array)
        plotargs = (
                fignames,
                figpaths,
                model,
                dataloader,
                device,
                config.fname_dataset,
                z_array_AEC,
                z_array,
                labels_GMM,
                labels,
                centroids_GMM,
                centroids,
                tsne_results,
                0,
                config.show
        )
        #plot_process = threading.Thread(
        #    target=plotting.plotter_mp,
        #    args=plotargs
        #)
        #plot_process.start()
        #print('complete.')

    elif config.model == 'AEC':

        running_loss = 0.
        running_size = 0

        for b, batch in enumerate(tqdm(dataloader)):
            #_, batch = batch
            x = batch.to(device)
            xr, z = model(x)
            loss = metrics[0](xr, x)

            z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
            xr_array[b * bsz:(b*bsz) + x.size(0), :] = xr.detach().cpu().numpy()

            running_loss += loss.cpu().detach().numpy() * x.size(0)
            running_size += x.size(0)

            pbar.set_postfix(
                {str(metrics)[0][:-2]: f"{(running_loss / running_size):.4e}"}
            )

        total_loss = running_loss / len(dataloader.dataset)
        print(f'Dataset MSE = {total_loss:.4e}')

        print('Saving data...', end="", flush=True)
        with open(os.path.join(savepath, 'MSE.txt'), 'w') as f:
            f.write(f'MSE = {total_loss:.4e}')
        np.save(os.path.join(savepath, 'Loss_AEC'), total_loss)
        np.save(os.path.join(savepath, 'Z_AEC'), z_array)
        np.save(os.path.join(savepath, 'Xr_AEC'), xr_array)
        print('complete.')


def model_training(config, model, dataloaders, metrics, optimizer, **hpkwargs):
    '''Primary machinery function for using AEC or DEC model in
    training mode.
    
    Parameters
    ----------
    config : Configuration object
        Object containing information about the experiment configuration
    
    model : PyTorch model instance
        Model with trained parameters.
    
    dataloaders : list
        List of PyTorch dataloader instances that load data from disk
        into memory.
    
    metrics : list
        List of PyTorch metric instances
    
    optimizer : PyTorch optimizer instance

    hpkwargs : dict
        Dictionary of hyperparameter values.
    '''

    def AEC_training(config, model, dataloaders, metrics, optimizer, tb, **hpkwargs):
        '''Subroutine for AEC training.
        
        Parameters
        ----------
        config : Configuration object
            Object containing information about the experiment configuration
        
        model : PyTorch model instance
            Model with trained parameters.
        
        dataloaders : list
            List of PyTorch dataloader instances that load data from disk
            into memory.
        
        metrics : list
            List of PyTorch metric instances
        
        optimizer : PyTorch optimizer instance

        tb : Tensorboard instance
            For writing results to Tensorboard.

        hpkwargs : dict
            Dictionary of hyperparameter values.
        '''
        batch_size = hpkwargs.get('batch_size')
        lr = hpkwargs.get('lr')
        device = config.device
        savepath_run = config.savepath_run
        savepath_chkpnt = config.savepath_chkpnt

        tra_loader = dataloaders[0]
        val_loader = dataloaders[1]

        if config.early_stopping:
            best_val_loss = 10000

        epochs = list()
        tra_losses = list()
        val_losses = list()
        finished = False
        n_epochs = config.n_epochs


        #fig = plotting.compare_images(
        #    model,
        #    0,
        #    config,
        #    savepath=savepath_run
        #)
        #tb.add_figure(
        #    'TrainingProgress',
        #    fig,
        #    global_step=0,
        #    close=True
        #)
        #del fig

        for epoch in range(n_epochs):

            print('-' * 100)
            print(
                f'Epoch [{epoch+1}/{n_epochs}] | '
                f'Batch Size = {batch_size} | LR = {lr}'
            )

            # ==== Training Loop: =============================================
            model, epoch_tra_mse = batch_training(model, tra_loader, optimizer, metrics[0], device)
            tb.add_scalar('Training MSE', epoch_tra_mse, epoch+1)

            #if ((epoch + 1) % 5) == 0 and not (epoch == 0):
            #    fig = plotting.compare_images(
            #        model,
            #        epoch + 1,
            #        config,
            #        savepath=savepath_run
            #    )
            #    tb.add_figure(
            #        'TrainingProgress',
            #        fig,
            #        global_step=epoch+1,
            #        close=True
            #    )
            #    del fig
#
            # ==== Validation Loop: ===========================================
            epoch_val_mse = batch_validation(model, val_loader, metrics, config)[0]

            tb.add_scalar('Validation MSE', epoch_val_mse, epoch+1)

            epochs, tra_losses, val_losses = utils.add_to_history(
                [epochs, tra_losses, val_losses],
                [epoch+1, epoch_tra_mse, epoch_val_mse]
            )

            if config.early_stopping:
                if epoch_val_mse < best_val_loss:
                    strikes = 0
                    best_val_loss = epoch_val_mse
                    torch.save(
                        model.state_dict(),
                        os.path.join(savepath_chkpnt, 'AEC_Best_Weights.pt')
                    )
                else:
                    if epoch == 0:
                        strikes = 1
                    else:
                        strikes += 1

                if epoch > config.patience and strikes > config.patience:
                    print('Stopping Early.')
                    finished = True
                    break
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(savepath_chkpnt, f'AEC_Params_{epoch+1:03d}.pt')
                )

        # Collect Results =====================================================
        hist_path = os.path.join(savepath_run, 'AEC_history.csv')
        _ = utils.save_history(
            {
                'Epoch': epochs,
                'Training Loss': tra_losses,
                'Validation Loss': val_losses
            },
            hist_path
        )

        #fig = plotting.view_history_AEC(hist_path)
        #fig.savefig(hist_path[:-4] + '.png', dpi=300, facecolor='w')
        #del fig
#
        #tb.add_hparams(
        #    {'Batch Size': batch_size, 'LR': lr},
        #    {
        #        'hp/Training MSE': epoch_tra_mse,
        #        'hp/Validation MSE': epoch_val_mse
        #    }
        #)

        #fig = plotting.compare_images(
        #    model,
        #    epoch+1,
        #    config,
        #    savepath=savepath_run
        #)
        #tb.add_figure(
        #    'TrainingProgress',
        #    fig,
        #    global_step=epoch+1,
        #    close=True
        #)
        fname = os.path.join(savepath_run, 'AEC_Params_Final.pt')
        if config.early_stopping and (finished == True or epoch == n_epochs-1):
            shutil.move(
                os.path.join(savepath_chkpnt, 'AEC_Best_Weights.pt'),
                fname
            )
        else:
            torch.save(model.state_dict(), fname)
        tb.add_text("Path to Saved Weights", fname, global_step=None)
        print('AEC parameters saved.')
        print(f'Path to saved weights: {fname}')


    def DEC_training(config, model, dataloaders, metrics, optimizer, tb, **hpkwargs):
        '''Subroutine for DEC training.
        
        Parameters
        ----------
        config : Configuration object
            Object containing information about the experiment configuration
        
        model : PyTorch model instance
            Model with trained parameters.
        
        dataloaders : list
            List of PyTorch dataloader instances that load data from disk
            into memory.
        
        metrics : list
            List of PyTorch metric instances
        
        optimizer : PyTorch optimizer instance

        tb : Tensorboard instance
            For writing results to Tensorboard.

        hpkwargs : dict
            Dictionary of hyperparameter values.
        '''
        batch_size = hpkwargs.get('batch_size')
        lr = hpkwargs.get('lr')
        n_clusters = hpkwargs.get('n_clusters')
        gamma = hpkwargs.get('gamma')
        tol = hpkwargs.get('tol')
        device = config.device
        savepath_run = config.savepath_run

        tra_loader = dataloaders[0]

        model.double()

        fignames = [
            'T-SNE',
            'Gallery',
            'LatentSpace',
            'CDF',
            'PDF'
        ]
        figpaths = [os.path.join(savepath_run, name) for name in fignames]
        [os.makedirs(path, exist_ok=True) for path in figpaths]

        model.load_state_dict(
            torch.load(config.saved_weights, map_location=device), strict=False
        )
        model.eval()

        metric_mse = metrics[0]
        metric_kld = metrics[1]

        M = len(tra_loader.dataset)
        if config.update_interval == -1:
            update_interval = int(np.ceil(M / (batch_size * 2)))
        else:
            update_interval = int(np.ceil(M / (batch_size * config.update_interval)))

        tb = SummaryWriter(log_dir = savepath_run)
        if config.tbpid is not None:
            tb.add_text(
                "Tensorboard PID",
                f"To terminate this TB instance, kill PID: {config.tbpid}",
                global_step=None
            )
        tb.add_text("Path to Saved Outputs", savepath_run, global_step=None)

        labels_prev, centroids = initialize_clusters(
            model,
            tra_loader,
            config,
            n_clusters=n_clusters
        )
        cluster_weights = torch.from_numpy(centroids).to(device)
        with torch.no_grad():
            model.state_dict()["clustering.weights"].copy_(cluster_weights)
        torch.save(
            model.state_dict(),
            os.path.join(savepath_run, 'DEC_Params_Initial.pt')
        )
        print('complete.')

        q, _, z_array0 = batch_eval(tra_loader, model, device) # <-- The CUDA problem occurs in here
        p = target_distribution(q)
        epoch = 0

        tsne_results = tsne(z_array0)

        plotargs = (
                fignames,
                figpaths,
                model,
                tra_loader,
                device,
                config.fname_dataset,
                z_array0,
                z_array0,
                labels_prev,
                labels_prev,
                centroids,
                centroids,
                tsne_results,
                epoch,
                config.show
        )
        plotkwargs = {
            'tb': tb
        }
        #plot_process = threading.Thread(
        #    target=plotting.plotter_mp,
        #    args=plotargs,
        #    kwargs=plotkwargs
        #)
        #plot_process.start()

        iters = list()
        rec_losses = list()
        clust_losses = list()
        total_losses = list()

        deltas_iter = list()
        deltas = list()

        n_iter = 1
        n_epochs = config.n_epochs
        finished = False

        for epoch in range(n_epochs):
            print('-' * 100)
            print(
                f'Epoch [{epoch+1}/{n_epochs}] | '
                f'# Clusters = {n_clusters} | '
                f'Batch Size = {batch_size} | '
                f'LR = {lr} | '
                f'gamma = {gamma} | '
                f'tol = {tol}'
            )

            running_loss = 0.0
            running_loss_rec = 0.0
            running_loss_clust = 0.0
            running_size = 0

            pbar = tqdm(
                tra_loader,
                leave=True,
                unit="batch",
                postfix={
                    "MSE": "%.6f" % 0.0,
                    "KLD": "%.6f" % 0.0,
                    "Loss": "%.6f" % 0.0
                },
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )

            # Iterate over data:
            for batch_num, batch in enumerate(pbar):
                #_, batch = batch
                x = batch.to(device)
                # Update target distribution, check performance
                if (batch_num % update_interval == 0) and not \
                    (batch_num == 0 and epoch == 0):
                    q, labels, _ = batch_eval(tra_loader, model, device)
                    p = target_distribution(q)
                    # check stop criterion
                    delta_label = np.sum(labels != labels_prev).astype(np.float32)\
                        / labels.shape[0]
                    deltas_iter, deltas = utils.add_to_history(
                        [deltas_iter, deltas],
                        [n_iter, delta_label]
                    )
                    # deltas.append(delta_label)
                    tb.add_scalar('delta', delta_label, n_iter)
                    labels_prev = np.copy(labels)
                    if delta_label < tol:
                        time.sleep(2)
                        print('Stop criterion met, training complete.')
                        finished = True
                        break

                tar_dist = p[running_size:(running_size + x.size(0)), :]
                tar_dist = torch.from_numpy(tar_dist).to(device)

                # zero the parameter gradients
                model.train()
                optimizer.zero_grad()

                # Calculate losses and backpropagate
                with torch.set_grad_enabled(True):
                    q, x_rec, _ = model(x)
                    loss_rec = metric_mse(x_rec, x)
                    loss_clust = gamma * metric_kld(torch.log(q), tar_dist) \
                        / x.size(0)
                    loss = loss_rec + loss_clust
                    loss.backward()
                    optimizer.step()

                running_size += x.size(0)
                running_loss += loss.detach().cpu().numpy() * x.size(0)
                running_loss_rec += loss_rec.detach().cpu().numpy() * x.size(0)
                running_loss_clust += loss_clust.detach().cpu().numpy() * x.size(0)

                accum_loss = running_loss / running_size
                accum_loss_rec = running_loss_rec / running_size
                accum_loss_clust = running_loss_clust / running_size

                pbar.set_postfix(
                    MSE = f"{accum_loss_rec:.4e}",
                    KLD = f"{accum_loss_clust:.4e}",
                    Loss = f"{accum_loss:.4e}"
                )

                iters, rec_losses, clust_losses, total_losses = \
                    utils.add_to_history(
                        [iters, rec_losses, clust_losses, total_losses],
                        [n_iter, accum_loss_rec, accum_loss_clust, accum_loss]
                    )
                tb.add_scalars(
                    'Losses',
                    {
                        'Loss': accum_loss,
                        'MSE': accum_loss_rec,
                        'KLD': accum_loss_clust
                    },
                    n_iter
                )
                tb.add_scalar('Loss', accum_loss, n_iter)
                tb.add_scalar('MSE', accum_loss_rec, n_iter)
                tb.add_scalar('KLD', accum_loss_clust, n_iter)

                n_iter += 1

            # Save figures every 4 epochs or at end of training ===============
            if (((epoch + 1) % 4 == 0) and not (epoch == 0)) or finished:
                _, _, z_array1 = batch_eval(tra_loader, model, device)
                tsne_results = tsne(z_array1)
                plotargs = (
                        fignames,
                        figpaths,
                        model,
                        tra_loader,
                        device,
                        config.fname_dataset,
                        z_array0,
                        z_array1,
                        labels_prev,
                        labels,
                        centroids,
                        model.clustering.weights.detach().cpu().numpy(),
                        tsne_results,
                        epoch+1,
                        config.show
                )
                plotkwargs = {'tb': tb}
                #plot_process = threading.Thread(
                #    target=plotting.plotter_mp,
                #    args=plotargs,
                #    kwargs=plotkwargs
                #)
                #plot_process.start()

            if finished:
                break

        _ = utils.save_history(
            {
                'Iteration': iters,
                'Reconstruction Loss': rec_losses,
                'Clustering Loss': clust_losses,
                'Total Loss': total_losses
            },
            os.path.join(savepath_run, 'DEC_history.csv')
        )
        _ = utils.save_history(
            {
                'Iteration': deltas_iter,
                'Delta': deltas
            },
            os.path.join(savepath_run, 'Delta_history.csv')
        )
        tb.add_hparams(
            {
                'Clusters': n_clusters,
                'Batch Size': batch_size,
                'LR': lr,
                'gamma': gamma,
                'tol': tol},
            {
                'hp/MSE': accum_loss_rec,
                'hp/KLD': accum_loss_clust,
                'hp/Loss': accum_loss
            }
        )

        fname = os.path.join(savepath_run, 'DEC_Params_Final.pt')
        torch.save(model.state_dict(), fname)
        tb.add_text("Path to Saved Weights", fname, global_step=None)
        tb.close()
        print('DEC parameters saved.')


    tic = datetime.now()
    print('Commencing training...')

    tb = SummaryWriter(log_dir=config.savepath_run)
    if config.tbpid is not None:
        tb.add_text(
            "Tensorboard PID",
            f"To terminate this TB instance, kill PID: {config.tbpid}",
            global_step=None
        )
    tb.add_text("Path to Saved Outputs", config.savepath_run, global_step=None)

    if config.model == "AEC":
        AEC_training(
            config,
            model,
            dataloaders,
            metrics,
            optimizer,
            tb,
            **hpkwargs
        )
    elif config.model == "DEC":
        DEC_training(
            config,
            model,
            dataloaders,
            metrics,
            optimizer,
            tb,
            **hpkwargs
        )

    toc = datetime.now()
    print(f'Training complete at {toc}; time elapsed = {toc-tic}.')


def silhouette_samples_X(x, labels, RF=2):
    '''Calculates silhouette scores for the data space, i.e.,
    spectrograms. Because of memory constraints, the silhouette score of
    the entire dataset of spectrograms cannot be calculated, so the
    data space is decimated by a reduction factor (RF). A GPU-enabled
    score is computed if CUDA is available.

    Parameters
    ----------
    x : array (M, D_)
        Data space, i.e., spectrograms (M samples, D_ features)

    Returns
    -------
    scores : array (M / RF,)
        Array containing sample-wise silhouette scores.

    x_ : array (M / RF, D_)
        Samples data space, i.e., spectrograms (M / RF samples, 
        D_ features)
    '''
    x_ = x[:, :, ::int(RF), ::int(RF)].squeeze()
    _, n, o = x_.shape
    x_ = np.reshape(x_, (-1, n * o))
    scores = silhouette_samples(x_, labels, chunksize=20000)
    if torch.cuda.is_available():
        scores = cupy.asnumpy(scores)
    x_ = np.reshape(x_, (-1, n, o))
    return scores, x_


def target_distribution(q):
    '''From Xie/Girshick/Farhadi (2016). Computes the target distribution p,
    given soft assignements, q. The target distribtuion is generated by giving
    more weight to 'high confidence' samples - those with a higher probability
    of being a signed to a certain cluster.  This is used in the KL-divergence
    loss function.

    Parameters
    ----------
    q : array (M,D)
        Soft assignement probabilities - Probabilities of each sample being
        assigned to each cluster [n_samples, n_features]

    Returns
    -------
    p : array (M,D)
        Auxiliary target distribution of shape [n_samples, n_features].
    '''
    p = q ** 2 / np.sum(q, axis=0)
    p = np.transpose(np.transpose(p) / np.sum(p, axis=1))
    return np.round(p, 5)


def tsne(data):
    '''Perform t-SNE on data.

    Parameters
    ----------
    data : array (M,N)

    Returns
    -------
    results : array (M,2)
        2-D t-SNE embedding
    '''
    print('Running t-SNE...', end="", flush=True)
    M = len(data)
    np.seterr(under='warn')
    results = TSNE(
        n_components=2,
        perplexity=int(M/100),
        early_exaggeration=20,
        learning_rate=int(M/12),
        n_iter=2000,
        verbose=0,
        random_state=2009
    ).fit_transform(data.astype('float32'))
    print('complete.')
    return results