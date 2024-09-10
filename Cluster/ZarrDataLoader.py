import torch
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
import dask.array as da
import zarr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ZarrDataset(Dataset):
    class SpecgramNormalizer(object):
        def __init__(self, transform=None):
            self.transform = transform

        def __call__(self, X):
            X = X.to(torch.float32)  # Ensure the tensor is floating-point for division
            if self.transform == "sample_normalization":
                X /= torch.abs(X).amax(dim=(1, 2), keepdim=True)
            elif self.transform == "sample_norm_cent":
                X = (X - X.mean()) / (torch.abs(X).amax() + 1e-8)
            elif self.transform == "vec_norm":
                shape = X.shape
                X = X.view(shape[0], -1)
                norm = torch.linalg.norm(X, dim=1, keepdim=True) + 1e-8
                X /= norm
                X = X.view(*shape)
            return X

    class SpecgramCrop(object):
        def __call__(self, X):
            return X[:-1, 1:]

    class SpecgramToTensor(object):
        def __call__(self, X):
            #print('initial shape: ', X.shape)
            X = np.expand_dims(X, axis=0)
            return torch.from_numpy(X)

    def __init__(self, zarr_path, sample_size, transform=None):
        # Open the Zarr dataset as an xarray Dataset, this will handle lazy loading
        #self.ds = xr.open_zarr(zarr_path, consolidated=True)
        #zarr_array = zarr.open(zarr_path, mode='r')
        group = zarr.open_group(zarr_path, mode='r')
        zarr_array = group[list(group.keys())[0]]
        self.ds = da.from_zarr(zarr_array)
        self.chunk_size = 5758

        self.sample_size = sample_size  # Size of each individual sample in the 'time' dimension
        self.transform = transform

        # Assuming each sample is non-overlapping for simplicity
        #self.num_samples = self.ds.dims['time'] // sample_size * self.ds.dims['channel']
        self.num_samples = (self.ds.shape[0] // 11 * 2) // self.chunk_size * ((self.ds.shape[1] - 1600) // 5)
        self.spectrogram_size = 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_time = (idx * self.chunk_size) % (self.ds.shape[0] // 11 * 2)
        end_time = start_time + self.chunk_size

        channel = (idx * self.chunk_size) // (self.ds.shape[0] // 11 * 2) * 5 + 1600

        # Load the entire chunk
        chunk = torch.from_numpy(self.ds[start_time:end_time, channel, :].compute()).double()

        # Split the chunk into spectrograms
        spectrograms = [chunk[i:i + self.spectrogram_size, :] for i in range(0, len(chunk), self.spectrogram_size)]

        # Process each spectrogram
        processed_spectrograms = []
        for spec in spectrograms:
            if self.transform is not None:
                spec = self.transform(spec)
            spec = spec.unsqueeze(0)
            if spec.shape[1] == 4:
                processed_spectrograms.append(spec)

        # Stack the spectrograms into a batch
        batch = torch.stack(processed_spectrograms)
        return batch



def get_zarr_data(split_dataset=True):
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        lambda x: x.double(),
    ])

    sample_size = 4
    full_dataset = ZarrDataset('/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_5758.zarr', sample_size, transform=transform_pipeline)
    #full_dataset = ZarrDataset("/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_60.zarr", sample_size, transform=transform_pipeline)
    print('full dataset length: ', len(full_dataset))

    if split_dataset:

        # Determine the size of the training and test sets
        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Split the dataset into training and test sets
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


        return train_dataset, test_dataset
    else:
        return full_dataset


