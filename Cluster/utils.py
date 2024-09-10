#!/usr/bin/env python3

"""William Jenkins
Scripps Institution of Oceanography, UC San Diego
wjenkins [at] ucsd [dot] edu
January 2021

Utility and helper functions for Cluster.
"""

import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import configparser
import csv
from datetime import datetime
import json
import os
import pickle
import re
from shutil import copyfile
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import transforms
from tqdm import tqdm


class Configuration():

    def __init__(self, init_path):
        self.init_path = init_path
        self.datatype = {
            'n_epochs': 'int',
            'show': 'bool',
            'send_message': 'bool',
            'early_stopping': 'bool',
            'patience': 'int',
            'km_metrics': 'bool',
            'tb': 'bool',
            'tbport': 'int',
            'workers': 'int',
            'batch_size': 'int',
            'lr': 'float',
            'n_clusters': 'int',
            'gamma': 'float',
            'tol': 'float',
            'update_interval': 'int'
        }


    def calc_tuning_runs(self):
        self.runs = 1
        for k in self.hp:
            self.runs *= len(self.hp[k])
        return self.runs


    def init_exp_env(self):
        self.serial_exp = datetime.now().strftime('%Y%m%dT%H%M%S')
        if self.mode == 'train':
            self.savepath_exp = os.path.join(self.savepath, 'Models', self.model, f'Exp{self.serial_exp}')
        elif self.mode == 'predict':
            fname = os.path.abspath(os.path.join(self.saved_weights, os.pardir))
            self.savepath_exp = os.path.join(fname, 'Prediction')
        elif self.mode == 'fit':
            fname = os.path.abspath(os.path.join(self.saved_weights, os.pardir))
            self.savepath_exp = os.path.join(fname, 'GMM')
        else:
            raise ValueError(
                'Wrong mode selected; choose "pretrain" or "train".'
            )
        os.makedirs(self.savepath_exp, exist_ok=True)

        print('New experiment file structure created at:\n'
              f'{self.savepath_exp}')
        return self.savepath_exp, self.serial_exp


    def init_output_env(self, **kwargs):
        self.serial_run = datetime.now().strftime('%Y%m%dT%H%M%S')
        if self.mode == 'train':
            if self.model == 'AEC':
                fname = f'Run_BatchSz={kwargs.get("batch_size")}_' + \
                        f'LR={kwargs.get("lr")}'
            elif self.model == 'DEC':
                fname = f'Run_Clusters={kwargs.get("n_clusters")}_' + \
                        f'BatchSz={kwargs.get("batch_size")}_' + \
                        f'LR={kwargs.get("lr")}_' + \
                        f'gamma={kwargs.get("gamma")}_' + \
                        f'tol={kwargs.get("tol")}'
            self.savepath_run = os.path.join(self.savepath_exp, fname)
            self.savepath_chkpnt = os.path.join(self.savepath_run, 'tmp')
            os.makedirs(self.savepath_run, exist_ok=True)
            os.makedirs(self.savepath_chkpnt, exist_ok=True)
            return self.serial_run, self.savepath_run, self.savepath_chkpnt

        elif self.mode == 'predict':
            pass
        elif self.mode == 'fit':
            fname = f'n_clusters={kwargs.get("n_clusters")}'
            self.savepath_run = os.path.join(self.savepath_exp, fname)
            os.makedirs(self.savepath_run, exist_ok=True)
            return self.serial_run, self.savepath_run

        #     n_clusters = kwargs.get('n_clusters')
        #     with open(f'{savepath}{n_clusters}_Clusters', 'w') as f:
        #         pass
        #     savepath_run = []
        #     for label in range(n_clusters):
        #         savepath_cluster = f'{savepath}Cluster{label:02d}'
        #         if not os.path.exists(savepath_cluster):
        #             os.makedirs(savepath_cluster)
        #         savepath_run.append(savepath_cluster)
        #     return savepath_run, serial_run
        # else:
        #     raise ValueError(
        #             'Wrong mode selected; choose "pretrain", "train", or "predict".'
        #         )


    def load_config(self):
        self.config = configparser.ConfigParser()
        self.config.read_file(open(self.init_path))
        self.section = self.config.sections()
        self.hp = dict()

        for section in self.section:
            for k, v in self.config.items(section):

                if section == "HYPERPARAMETERS":
                    if self.datatype[k] == "int":
                        v=[int(i) for i in v.split(', ')]
                    elif self.datatype[k] == "float":
                        v=[float(i) for i in v.split(', ')]
                    self.hp[k] = v

                else:
                    if k in self.datatype.keys():
                        if self.datatype[k] == "int":
                            v = int(v)
                        elif self.datatype[k] == "bool":
                            v = self.config[section].getboolean(k)

                    if k == "img_index":
                        v = [int(i) for i in v.split(',')]
                    if k == "klist":
                        v = np.arange(int(v.split(',')[0]), int(v.split(',')[1])+1)

                    setattr(self, k, v)

        if self.mode == 'predict':
            self.verb = 'prediction'
            if self.model == 'DEC':
                self.n_clusters = int(parse_nclusters(self.saved_weights))
            else:
                self.n_clusters = None
        elif self.mode == 'train':
            self.verb = 'training'
        elif self.mode == 'fit':
            self.verb = 'fitting'
        self.tbpid = None

        if self.model == 'AEC' and len(self.hp.keys()) > 2:
            self.hp = {k: v for k, v in self.hp.items() if (k == 'batch_size') or (k == 'lr')}
        self.runs = self.calc_tuning_runs()


    def load_TraVal_index(self):
        with open(self.indexpath, 'rb') as f:
            data = pickle.load(f)
            self.index_tra = data['index_tra']
            self.index_val = data['index_val']
        return self.index_tra, self.index_val


    def save_exp_config(self):
        fname = os.path.join(self.savepath_exp, f'ExpConfig{self.serial_exp}')
        copyfile(self.init_path, f'{fname}.ini')
        with open(f'{fname}.txt', 'w') as f:
            f.write(str(self.__dict__))
        with open(f'{fname}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)


    def set_device(self, cuda_device=None):
        if torch.cuda.is_available():
            if cuda_device is not None:
                self.device = torch.device(f'cuda:{cuda_device}')
            else:
                self.device = torch.device('cuda')
            print(f'CUDA device available, using GPU ({torch.cuda.current_device()}).')
        else:
            self.device = torch.device('cpu')
            print('CUDA device not available, using CPU.')
        return self.device


    def start_tensorboard(self):
        cmd = f"python -m tensorboard.main --logdir=. --port={self.tbport} --samples_per_plugin images=1000"
        p = subprocess.Popen(cmd, cwd=self.savepath_exp, shell=True)
        self.tbpid = p.pid
        print(f"Tensorboard server available at http://localhost:{self.tbport}; PID={self.tbpid}")
        return self.tbpid


class SeismicDataset(Dataset):

    class SpecgramNormalizer(object):

        def __init__(self, transform=None):
            self.transform = transform


        def __call__(self, X):
            n, o = X.shape
            if self.transform == "sample_normalization":
                X /= np.abs(X).max(axis=(0,1))
            elif self.transform == "sample_norm_cent":
                X = (X - X.mean()) / np.abs(X).max()
            elif self.transform == "vec_norm":
                X = np.reshape(X, (1,-1))
                X /= np.linalg.norm(X)
                X = np.reshape(X, (n, o))

            return X


    class SpecgramCrop(object):
        """Crop & reshape data."""
        def __call__(self, X):
            return X[:-1, 1:]


    class SpecgramToTensor(object):
        """Convert ndarrays in sample to Tensors."""
        def __call__(self, X):
            X = np.expand_dims(X, axis=0)
            return torch.from_numpy(X)

    def __init__(
            self,
            fname,
            ftype,
            transform=transforms.Compose(
                [
                    SpecgramCrop(),
                    SpecgramNormalizer(transform='vec_norm'),
                    SpecgramToTensor()
                ]
            )
        ):
        self.transform = transform
        self.fname = fname
        self.ftype = ftype
        if self.ftype == 'np':
            self.data = np.load(self.fname)


    def __len__(self):
        if self.ftype == 'np':
            return self.data.shape[0]
        elif self.ftype == 'h5':
            m, _, _ = query_dbSize(self.fname)
            return m


    def __getitem__(self, idx):
        if self.ftype == 'np':
            X = torch.from_numpy(self.data[idx,:])
        elif self.ftype == 'h5':
            X = torch.from_numpy(self.read_h5(self.fname, idx))
        if self.transform:
            X = self.transform(X)
        return idx, X


    @staticmethod
    def read_h5(fname, idx):
        with h5py.File(fname, 'r') as f:
            DataSpec = '/4.0/Spectrogram'
            return f[DataSpec][idx]


class LabelCatalogue(object):

    def __init__(self, paths, label_list=None, threshold=None):
        self.paths = paths
        self.freq = None
        self.df = self.build_df(self.paths)
        if label_list is not None:
            self.label_list = label_list.sort()
        else:
            self.label_list = np.sort(pd.unique(self.df["label"]))
        self.station_list = pd.unique(self.df["station"])
        if threshold is not None:
            self.threshold = threshold
            self.apply_threshold()


    def amplitude_statistics(self):
        columns = ["Class", "Mean", "Median", "Standard Deviation", "Maximum"]
        stats = []
        for label in self.label_list:
            subset = self.df["peak"].loc[self.df["label"] == label].abs()
            stats.append(
                (
                    label+1,
                    subset.mean(),
                    subset.median(),
                    subset.std(),
                    subset.max()
                )
            )

        amp_stats = pd.DataFrame(
            stats,
            columns=columns
        ).sort_values(by=["Class"], ignore_index=True)
        return amp_stats.set_index("Class")

    
    def apply_threshold(self, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        if isinstance(self.threshold, float):
            print(self.threshold)
            self.df = self.df[self.df['peak'] >= self.threshold]
        elif isinstance(self.threshold, list):
            if len(self.threshold) != 2:
                raise ValueError('Threshold requires 1 or 2 values!')

            lo = min(self.threshold)
            if lo == 0:
                lo -= 1 # Catches values ~0 missed due to floating point error
            hi = max(self.threshold)
            self.df = self.df[(self.df['peak'] >= lo) & (self.df['peak'] < hi)]

        return self.df


    def build_df(self, paths):
        data1 = pd.read_csv(paths[0]).drop(columns=["Index"])
        data2 = pd.read_csv(self.paths[1])
        df = pd.concat(
            [data1, data2],
            axis=1
        ).drop(
            columns=[
                "channel",
                "dt_on",
                "dt_off",
                "fs",
                "delta",
                "npts",
                "STA",
                "LTA",
                "on",
                "off",
                "spec_start",
                "spec_stop"
            ]
        ).rename(columns={"dt_peak": "time"})
        df["time"] = df["time"].astype("datetime64")
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df


    def gather_counts(self, station, freq="month", label_list=None):
        if freq == "month":
            freqcode = "1M"
        elif freq == "day":
            freqcode = "1D"
        elif freq == "hour":
            freqcode = "1H"
        self.freq = freq
        if (label_list is not None) and \
            (max(label_list) > max(self.label_list)):
            raise ValueError("label_list includes impossibly high label.")
        else:
            label_list = self.label_list
        for i, label in enumerate(label_list):
            mask = (self.df["station"] == station) & \
                (self.df['label'] == label)
            subset = self.df.loc[mask].drop(
                columns = ["network", "station", "peak", "unit", "idx"]
            )
            counts = subset.resample(freqcode).count().rename(
                columns={"label": f"{label+1}"}
            )
            if i == 0:
                df = counts
            else:
                df = pd.concat([df, counts], axis=1)
        return df.fillna(0).astype("int").sort_index()


    def get_peak_freq(self, fname_dataset, batch_size=2048, workers=12):

        _, _, fvec = load_images(fname_dataset, [[0]])

        dataset = SeismicDataset(fname_dataset, 'h5')

        class_avg_maxfreq = np.zeros(len(self.label_list))
        for j, label in enumerate(self.label_list):
            mask = self.df.label == label
            labels_subset = self.df.loc[mask]

            subset = Subset(dataset, labels_subset.idx)
            dataloader = DataLoader(
                subset,
                batch_size=batch_size,
                num_workers=workers
            )
            batch_avg_maxfreq = np.zeros(len(dataloader))
            pbar = tqdm(
                dataloader,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                desc=f"Label {j+1}/{len(self.label_list)} (Class {label})",
                leave=True,
                position=0
            )
            for i, batch in enumerate(pbar):
                _, X = batch
                maxfreqind = torch.max(torch.sum(X, 3) / X.size(3), 2).indices
                maxfreqind = maxfreqind.detach().cpu().numpy()
                maxfreq = fvec[maxfreqind]
                batch_avg_maxfreq[i] = maxfreq.mean()

            class_avg_maxfreq[j] = batch_avg_maxfreq.sum() / len(dataloader)
            print(
                f"Avg. Peak Frequency: {class_avg_maxfreq[j]:.2f} Hz",
                flush=True
            )

        peak_freqs = pd.DataFrame(
            {"Class": self.label_list, "Avg_Peak_Freq": class_avg_maxfreq}
        ).sort_values(by=["Class"], ignore_index=True)
        return peak_freqs.set_index("Class")

    
    @staticmethod
    def load_images(fname_dataset, index):
        with h5py.File(fname_dataset, 'r') as f:
            #samples, frequency bins, time bins, amplitude
            DataSpec = '/4.0/Spectrogram'
            dset = f[DataSpec]
            X = np.zeros((len(index), 1201, 2301))
            for i, index in enumerate(index):
                dset_arr = dset[index, :, :]
                X[i] = dset_arr
            fvec = dset[0, 0:87, 0]
            tvec = dset[0, 87, 1:]
        X = X[:, :-1, 1:]

        X = (X - X.mean(axis=(1,2))[:,None,None]) / \
            np.abs(X).max(axis=(1,2))[:,None,None]

        X = np.expand_dims(X, axis=1)
        return X, tvec, fvec
    
    
    def seasonal_statistics(self, mode=None):
        if mode is not None:
            count_summer15 = np.empty((len(self.label_list),))
            count_winter15 = np.empty((len(self.label_list),))
            count_summer16 = np.empty((len(self.label_list),))
            count_winter16 = np.empty((len(self.label_list),))
            total = np.empty((len(self.label_list),))

            for j, label in enumerate(self.label_list):
                mask_label = self.df["label"] == label
                subset = self.df.loc[mask_label]
                total_count = len(subset.index)
                mask_summer15 = (subset.index >= datetime(2015,1,1)) & \
                    (subset.index < datetime(2015,4,1))
                mask_winter15 = (subset.index >= datetime(2015,6,1)) & \
                    (subset.index < datetime(2015,9,1))
                mask_summer16 = (subset.index >= datetime(2016,1,1)) & \
                    (subset.index < datetime(2016,4,1))
                mask_winter16 = (subset.index >= datetime(2016,6,1)) & \
                    (subset.index < datetime(2016,9,1))

                count_summer15[j] = 100 * len(subset.loc[mask_summer15].index) / total_count
                count_winter15[j] = 100 * len(subset.loc[mask_winter15].index) / total_count
                count_summer16[j] = 100 * len(subset.loc[mask_summer16].index) / total_count
                count_winter16[j] = 100 * len(subset.loc[mask_winter16].index) / total_count
                total[j] = total_count

            return pd.DataFrame({
                "total": total,
                "JFMTotal": count_summer15 + count_summer16,
                "JFM15": count_summer15,
                "JFM16": count_summer16,
                "JJATotal": count_winter15 + count_winter16,
                "JJA15": count_winter15,
                "JJA16": count_winter16
            })
        else:
            count_summer = np.empty((len(self.label_list),))
            count_winter = np.empty((len(self.label_list),))
            for j, label in enumerate(self.label_list):
                mask_label = self.df["label"] == label
                subset = self.df.loc[mask_label]
                total_count = len(subset.index)
                mask_summer = ((subset.index >= datetime(2015,1,1)) & (subset.index < datetime(2015,4,1))) | \
                    ((subset.index >= datetime(2016,1,1)) & (subset.index < datetime(2016,4,1)))
                mask_winter = ((subset.index >= datetime(2015,6,1)) & (subset.index < datetime(2015,9,1))) | \
                    ((subset.index >= datetime(2016,6,1)) & (subset.index < datetime(2016,9,1)))

                count_winter[j] = 100 * len(subset.loc[mask_winter].index) / total_count
                count_summer[j] = 100 * len(subset.loc[mask_summer].index) / total_count
            return pd.DataFrame({"JFM": count_summer, "JJA": count_winter})


    def station_statistics(self):
        count = np.empty((len(self.station_list),))
        percent = np.empty((len(self.station_list),))
        total_count = len(self.df.index)
        label_matrix = np.empty((len(self.station_list),len(self.label_list)))
        for i, station in enumerate(self.station_list):
            mask_station = self.df["station"] == station
            subset_station = self.df.loc[mask_station]
            count[i] = len(subset_station.index)
            percent[i] = 100 * len(subset_station.index) / total_count
            for j, label in enumerate(self.label_list):
                mask_label = subset_station["label"] == label
                subset_label = subset_station.loc[mask_label]
                label_matrix[i, j] = len(subset_label.index)

        df = pd.DataFrame(
            {
                "station": self.station_list,
                "N": count,
                "percent": percent
            }
        )
        df = pd.concat([df, pd.DataFrame(label_matrix)], axis=1)
        for col in [1, range(3, 3 + len(self.label_list))]:
            df.iloc[:, col] = df.iloc[:, col].astype("int")
        return df.sort_values(by="station", ignore_index=True)


def add_to_history(ll, lv):
    """Appends values from a list of values to a list from a list of lists.

    Parameters
    ----------
    ll : list
        List of lists to which values will be appended.

    lv : list
        List of values to append to lists.

    Returns
    -------
    list
        List of lists to which values have been appended.
    """
    [ll[i].append(v) for i, v in enumerate(lv)]
    return [l for l in ll]


def config_training(universal, parameters, hp=None):
    """Takes parameter dictionaries, formats them into CONFIGPARSER
    format, and saves to disk.

    Parameters
    ----------
    universal : dict
        Contains workflow parameters common to all model modes.
    parameters : dict
        Contains workflow parameter specific to particular model mode.
    hp : dict
        Contains hyperparameters.
    
    Returns
    -------
    fname : str
        Path to configuration file.
    """
    config = configparser.ConfigParser()
    config['UNIVERSAL'] = universal
    config['PARAMETERS'] = parameters
    if hp is not None:
        config['HYPERPARAMETERS'] = hp
    fname = os.path.join(
        universal['configpath'],
        f'init_{parameters["mode"]}.ini'
    )
    with open(fname, 'w') as configfile:
        config.write(configfile)
    return fname


def dataset_to_RAM(dataset):
    """Stores dataset in RAM. For smaller datasets that can fit in CPU 
    memory, this allows for faster loading and processing times during
    model training and evaluation.

    Parameters
    ----------
    dataset : Dataset Object
        Pytorch Dataset instance.
    
    Returns
    -------
        Pytorch TensorDataset object.
    """
    bsz = 4096
    dl = DataLoader(dataset, batch_size=bsz, pin_memory=True)
    pbar = tqdm(
        dl,
        leave=True,
        desc="Loading Data",
        unit="batch",
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    m = dataset.__len__()
    _, n, o, p = next(iter(dl))[1].shape
    index = torch.zeros(m, dtype=torch.int)
    data = torch.zeros(m, n, o, p)

    for b, batch in enumerate(pbar):
        index[b * bsz:(b*bsz) + len(batch[0])], data[b * bsz:(b * bsz) + len(batch[1]),:] = batch

    return TensorDataset(index, data)


def distance_matrix(x, y, f):
    assert len(x) == len(y)
    M = len(x)
    dist = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            dist[i, j] = fractional_distance(
                x[np.newaxis,i],
                y[np.newaxis,j],
                f
            )
    return dist


def convert_H5_to_NP():
    parser = argparse.ArgumentParser(
        description="Converts spectrograms in H5 file to Numpy file."
    )
    parser.add_argument('source', help='Path to source dataset.')
    parser.add_argument('--dest', help='Path to destination.')
    parser.add_argument('--workers', help='Number of CPU workers.', nargs='?', const=1, default=1, type=int)
    args = parser.parse_args()

    bsz = 4096

    print(f'Loading data from {args.source}')
    dataset = SeismicDataset(args.source, 'h5')

    dataloader = DataLoader(
        dataset,
        batch_size=bsz,
        num_workers=args.workers,
        pin_memory=True
    )

    pbar = tqdm(
        dataloader,
        leave=True,
        desc="Loading",
        unit="batch",
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    x_array = np.zeros((len(dataloader.dataset), 1, 87, 100), dtype=np.float32)

    for b, batch in enumerate(tqdm(dataloader)):
        _, x = batch
        x_array[b * bsz:(b*bsz) + x.size(0), :] = x.detach().cpu().numpy()

    time.sleep(1)
    print('Saving data...', end='', flush=True)

    if args.dest is not None:
        dest = args.dest
    else:
        dest = args.source

    np.save(dest, x_array)
    print('complete.')
    return


def extractH5dataset():
    """Command line function that extracts a subset of an HDF database and
    saves it to a new HDF database.

    Parameters
    ----------
    source : str
        Path to HDF file from which new dataset will be extracted.

    dest : str
        Path to new HDF file

    include : str (optional)
        List of stations to extract

    exclude : str (optional)
        List of stations to exclude during extraction

    after : str (optional)
        Datetime after which to include (in format YYYYMMDDTHHMMSS)

    before : str (optional)
        Datetime before which to include (in format YYYYMMDDTHHMMSS)
    """
    def _copy_attributes(in_object, out_object):
        """Copy attributes between 2 HDF5 objects.

        Parameters
        ----------
        in_object : object
            Object to be copied

        out_ojbect : object
            Object to which in_object attributes are copied.
        """
        for key, value in in_object.attrs.items():
            out_object.attrs[key] = value


    def _find_indeces(index, source, stations):
        """Returns index of sample if it matches station input.

        Parameters
        ----------
        index : int
            Sample index to be queried.

        source : str
            Path to H5 database.

        stations : list
            List of stations to match.

        Returns
        -------
        index : int
            Sample index if match; nan otherwise.
        """
        with h5py.File(source, 'r') as f:
            metadata = json.loads(f['/4.0/Catalogue'][index])
        if metadata["Station"] in stations:
            return index
        else:
            return np.nan

    parser = argparse.ArgumentParser(
        description="Creates new dataset from existing dataset."
    )
    parser.add_argument("source", help="Enter path to source dataset.")
    parser.add_argument("dest", help="Enter path to destination dataset.")
    parser.add_argument("--include", help="Enter stations to include.")
    parser.add_argument("--exclude", help="Enter stations to exclude.")
    parser.add_argument("--after", help="Include after YYYYMMDDTHHMMSS.")
    parser.add_argument("--before", help="Include before YYYYMMDDTHHMMSS.")
    args = parser.parse_args()

    if args.include is None and args.exclude is None:
        raise Exception("Must specify stations to include or exclude.")
    source = args.source
    dest = args.dest
    include = args.include
    exclude = args.exclude
    after = args.after
    before = args.before

    if not os.path.exists(source):
        raise ValueError(f"Source file not found: {source}")

    with h5py.File(source, 'r') as rf:
        M = len(rf['/4.0/Trace'])
    index = np.arange(1, M)

    if include is not None:
        include = json.loads(include)
    else:
        include = None
    if exclude is not None:
        exclude = json.loads(exclude)
    else:
        exclude = None
    if after is not None:
        after = after
    else:
        after = None
    if before is not None:
        before = before
    else:
        before = None

    if include is not None and exclude is not None:
        removals = [get_station(i) for i in exclude]
        stations = [get_station(i) for i in include]
        stations = [i for i in stations if i not in removals]
        print(f"Searching {stations}")
    elif include is not None:
        stations = [get_station(i) for i in include]
        print(f"Searching {stations}")
    elif exclude is not None:
        removals = [get_station(i) for i in exclude]
        stations = [get_station(i) for i in range(34)]
        stations = [i for i in stations if i not in removals]
        print(f"Searching {stations}")
    else:
        stations = [get_station(i) for i in range(34)]
        print(f"Searching {stations}")

    A = [{
            "index": index[i],
            "source": source,
            "stations": stations
        } for i in range(len(index))]

    index_keep = np.zeros(len(index))
    with ProcessPoolExecutor(max_workers=14) as exec:
        print("Finding detections that meet filter criteria...")
        futures = [exec.submit(_find_indeces, **a) for a in A]
        kwargs1 = {
            "total": int(len(index)),
            "bar_format": '{l_bar}{bar:20}{r_bar}{bar:-20b}',
            "leave": True
        }
        for i, future in enumerate(tqdm(as_completed(futures), **kwargs1)):
            index_keep[i] = future.result()
    index_keep = np.sort(index_keep[~np.isnan(index_keep)]).astype(int)

    with h5py.File(source, 'r') as fs, h5py.File(dest, 'w') as fd:
        M = len(index_keep)
        dset_names = ['Catalogue', 'Trace', 'Spectrogram', 'Scalogram']
        for dset_name in dset_names:
            group_path = '/4.0'
            dset = fs[f"{group_path}/{dset_name}"]
            dset_shape = dset.shape[1:]
            dset_shape = (M,) + dset_shape
            group_id = fd.require_group(group_path)
            dset_id = group_id.create_dataset(
                dset_name,
                dset_shape,
                dtype=dset.dtype,
                chunks=None
            )
            _copy_attributes(dset, dset_id)
            kwargs2 = {
                "desc": dset_name,
                "bar_format": '{l_bar}{bar:20}{r_bar}{bar:-20b}'
            }
            for i in tqdm(range(len(index_keep)), **kwargs2):
                dset_id[i] = dset[index_keep[i]]


def fractional_distance(x, y, f):
    diff = np.fabs(x - y) ** f
    dist = np.sum(diff, axis=1) ** (1 / f)
    return dist


def generate_sample_index():
    """Command line function that generates a uniformly random sample index.

    Parameters
    ----------
    M : int
        Number of samples

    path : str
        Path to h5 dataset

    savepath : str
        Path to save sample index.
    """
    parser = argparse.ArgumentParser(description='Enter sample number.')
    parser.add_argument(
        'M',
        metavar='M',
        type=int,
        help='Enter number of spectrograms to be used for training/validation.'
    )
    parser.add_argument(
        'path',
        metavar='path',
        help='Enter path to h5 dataset.'
    )
    parser.add_argument(
        'savepath',
        metavar='savepath',
        help='Enter savepath'
    )
    args = parser.parse_args()
    M = args.M
    fname_dataset = args.path
    savepath = args.savepath
    save_TraVal_index(M, fname_dataset, savepath)


def get_channel(channel_index):
    '''Input: Integer channel index (0-2).
       Output: Channel name (str)'''
    channel_list = ['HHE', 'HHN', 'HHZ']
    channel_name = channel_list[channel_index]
    return channel_name


def get_datetime(datetime_index):
    '''Input: Integer datetime index for any day between ti and tf.
       Output: Datetime string'''
    ti = "20141202T000000"
    tf = "20161129T000000"
    datetimes = pd.date_range(ti, tf, freq='d')
    datetime = datetimes[datetime_index]
    return datetime


def get_metadata(query_index, sample_index, fname_dataset):
    '''Returns station metadata given sample index.'''
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4.0/Catalogue'
        dset = f[DataSpec]
        metadata = dict()
        counter = 0
        for i in query_index:
            query = sample_index[i]
            metadata[counter] = json.loads(dset[query])
            counter += 1
    return metadata


def get_network(network_index):
    '''Input: Integer network index (0).
       Output: Network name string'''
    network_list = ['XH']
    network_name = network_list[network_index]
    return network_name


def get_station(station):
    '''Returns station index or station name, depending on whether input is a
    name (string) or index (integer).

    Parameters
    ----------
    station : str, int
        Station name (str), Station index (int)

    Returns
    -------
    station: int, str
        Station index (int), Station name (str)
    '''
    station_list = ['DR01', 'DR02', 'DR03', 'DR04', 'DR05', 'DR06', 'DR07',
                   'DR08', 'DR09', 'DR10', 'DR11', 'DR12', 'DR13', 'DR14',
                   'DR15', 'DR16', 'RS01', 'RS02', 'RS03', 'RS04', 'RS05',
                   'RS06', 'RS07', 'RS08', 'RS09', 'RS10', 'RS11', 'RS12',
                   'RS13', 'RS14', 'RS15', 'RS16', 'RS17', 'RS18']
    if isinstance(station, int):
        return station_list[station]
    elif isinstance(station, str):
        return station_list.index(station)


def get_timefreqvec(fname_dataset):
    with h5py.File(fname_dataset) as f:
        DataSpec = '/4.0/Spectrogram'
        dset = f[DataSpec]
        tvec = dset[0, 87, 1:]
        fvec = dset[0, 0:87, 0]
    return tvec, fvec


def init_project_env(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"{path} created.")
        else:
            print(f"{path} exists.")
    print("Project folders initialized.")


# Move this to the LabelCatalogue class
def load_images(fname_dataset, index):
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/4.0/Spectrogram'
        dset = f[DataSpec]
        X = np.zeros((len(index), 1201, 2301))
        for i, index in enumerate(index):
            dset_arr = dset[index, :, :]
            X[i] = dset_arr
    #         X = dset_arr / np.abs(dset_arr).max()
    #         X = (dset_arr - dset_arr.mean()) / dset_arr.std()
        fvec = dset[0, 0:87, 0]
        tvec = dset[0, 87, 1:]
    X = X[:, :-1, 1:]

    X = (X - X.mean(axis=(1,2))[:,None,None]) / \
        np.abs(X).max(axis=(1,2))[:,None,None]

    X = np.expand_dims(X, axis=1)
    return X, tvec, fvec


def load_weights(model, fname, device):
    model.load_state_dict(torch.load(fname, map_location=device), strict=False)
    model.eval()
    print(f'Weights loaded to {device}')
    return model


def measure_class_inertia(data, centroids, n_clusters):
    inertia = np.empty(n_clusters)
    for j in range(n_clusters):
        mu = centroids[j]
        inertia[j] = np.sum(np.sqrt(np.sum((data - mu) ** 2, axis=1)) ** 2)
    return inertia


def measure_label_change(labels1, labels2):
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    label_list = np.unique(labels1)
    counts = np.zeros((len(label_list), len(label_list)), dtype=int)
    total_counts = np.zeros(len(label_list), dtype=int)
    percentages = np.zeros(len(label_list))

    M = len(labels1)
    for label in label_list:
        labels1_subset = labels1[labels1 == label]
        labels2_subset = labels2[labels1 == label]
        label_change_index = np.where(labels1_subset != labels2_subset)[0]
        new_labels = labels2_subset[label_change_index]
        counts_, _ = np.histogram(new_labels, bins=len(label_list))
        counts[label, :] = counts_
        total_counts[label] = counts_.sum()
        percentages[label] = 100 * counts_.sum() / M


    data = {'N_j': total_counts, '%N': percentages}
    data.update({f'Class {label+1}': counts[label,:] for label in label_list})
    df = pd.DataFrame(data=data)
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    df[[label for label in df.columns.values if label != '%N']] = df[[label for label in df.columns.values if label != '%N']].astype('int')
    df.index.names = ['Orig']

    return df


def parse_nclusters(line):
    """Do a regex search of PyTorch model and return the defined number of
    clusters in the model's clustering layer.

    Parameters
    ----------
    line : str
        PyTorch model clustering layer label.

    Returns
    -------
    int
        Number of clusters defined in model clustering layer.
    """
    rx_dict = {'n_clusters': re.compile(r'Clusters=(?P<n_clusters>\d+)')}
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return match.group('n_clusters')
        else:
            raise Exception('Unable to parse filename for n_clusters.')


def query_dbSize(path):
    """Queries the h5 database and returns data dimensions.

    Parameters
    ----------
    path : str
        Path to h5 database.

    Returns
    -------
    m, n, o : int
        Length, width, and height of data contained in h5 database.
    """
    with h5py.File(path, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/4.0/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        return m, n, o


def query_H5size():
    """Command line function that prints the dimensions of the specified HDF
    database.

    Parameters
    ----------
    path : str
        Path to the H5 database.
    """
    parser = argparse.ArgumentParser(description='Enter path to .h5 file.')
    parser.add_argument(
        'path',
        help='Enter path to database; must be .h5/.hd5 file.'
    )
    args = parser.parse_args()
    path = args.path
    m, n, o = query_dbSize(path)
    print(f" >> h5 dataset contains {m} samples with dimensions [{n},{o}]. <<")
    pass


def save_exp_config(savepath, serial, init_file, parameters, hyperparameters):
    """Saves experiment configuration to .txt and .pkl.

    Parameters
    ----------
    savepath : str
        Path to experiment folder.

    serial : str
        Experiment serial number.

    init_file : str
        Path to initialization configuration file.

    parameters : dict
        Dictionary containing experiment parameters.

    hyperparameters : dict
        Dictionary containing experiment hyperparameters.
    """
    fname = f'{savepath}ExpConfig{serial}'
    if hyperparameters is not None:
        configs = [parameters, hyperparameters]
    else:
        configs = parameters
    copyfile(init_file, f'{fname}.ini')
    with open(f'{fname}.txt', 'w') as f:
        f.write(str(configs))
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(configs, f)


def save_history(history, path):
    """Uses Pandas dataframe to write counter and scalars (epoch, iteration) to
    CSV file.

    Parameters
    ----------
    history : dict
        Dictionary of keys and values (array) to be saved. First dictionary
        item is used as the dataframe index.

    path : str
        Path to disk where CSV will be saved.

    Returns
    -------
    df : DataFrame
        DataFrame containing history values.
    """
    df = pd.DataFrame.from_dict(history).set_index(list(history.keys())[0])
    df.to_csv(path)
    return df


def save_labels(label_list, savepath, serial=None):
    """Writes clustering labels to CSV file.

    Parameters
    ----------
    label_list : dict
        Dictionary of keys and values (array) to be saved. First dictionary
        item is used as the dataframe index.

    savepath : str
        Path to disk where CSV will be saved.

    serial : str
        Experiment serial number.
    """
    if serial is not None:
        fname = os.path.join(savepath, f'Labels{serial}.csv')
    else:
        fname = os.path.join(savepath, 'Labels.csv')
    keys = label_list[0].keys()
    if not os.path.exists(fname):
        with open(fname, 'w') as csvfile:
            w = csv.DictWriter(csvfile, keys)
            w.writeheader()
            w.writerows(label_list)
    else:
        with open(fname, 'a') as csvfile:
            w = csv.DictWriter(csvfile, keys)
            w.writerows(label_list)


def save_TraVal_index(M, fname_dataset, savepath, reserve=0.0):


    def _set_TraVal_index(M, fname_dataset, reserve=0.0):
        with h5py.File(fname_dataset, 'r') as f:
            DataSpec = '/4.0/Spectrogram'
            m, _, _ = f[DataSpec].shape
            if M > m:
                print(f'{M} spectrograms requested, but only {m} '
                      f'available in database; setting M to {m}.')
                M = m
        index = np.random.choice(
            np.arange(1,m),
            size=int(M * (1+reserve)),
            replace=False
        )
        split_fraction = 0.8
        split = int(split_fraction * len(index))
        index_tra = index[0:split]
        index_val = index[split:]
        return index_tra, index_val, M


    index_tra, index_val, M = _set_TraVal_index(M, fname_dataset)
    index = dict(
        index_tra=index_tra,
        index_val=index_val
    )
    serial = datetime.now().strftime('%Y%m%dT%H%M%S')
    # savepath = f'{savepath}TraValIndex_M={M}_Res={reserve}_{serial}.pkl'
    savepath = f'{savepath}/TraValIndex_M={M}.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(index, f)
    print(f'{M} training & validation indices saved to:')
    print(savepath)
    return index_tra, index_val, savepath


def start_tensorboard(logdir, tbport):
    """Begins a new process for Tensorboard and returns its PID.

    Parameters
    ----------
    logdir : str
        Path to save Tensorbaord logs.

    tbport : int
        Port to which Tensorboard is attached.

    Returns
    -------
    tbpid : int
        PID of the Tensorboard process.
    """
    cmd = f"python -m tensorboard.main --logdir=. --port={tbport} --samples_per_plugin images=1000"
    p = subprocess.Popen(cmd, cwd=logdir, shell=True)
    tbpid = p.pid
    print(f"Tensorboard server available at http://localhost:{tbport}; PID={tbpid}")
    return tbpid
