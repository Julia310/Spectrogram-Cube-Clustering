from Cluster.production import train
from Cluster import utils
import os

# Main project folder to save outputs:
project_folder = '.'
# Path to configuration files:
path_config = f"{project_folder}/Config"
# Path to folder containing data, including HDF file for ML workflow:
path_data = f"{project_folder}/Data"
# Path to raw seismic data:
# path_data_seismo = f"{path_data}/Seismo"
path_data_seismo = "./zdata2/data/wfjenkin/RIS_Seismic"
# Path to save workflow outputs (ML models, figures, results, etc.)
path_output = f"{project_folder}/Outputs"
# Path to HDF dataset:
fname_dataset = f"{path_data}/RISData_20210713.h5"
# Path to save paper-ready figures:
figure_savepath = f"{path_output}/Figures"

os.makedirs('./Config/', exist_ok=True)
exp_name = "FullArray"

# Image Sample Indexes for Example Waveforms:
img_index = [1, 2, 3, 4]

# Generate new sample index for data set?
genflag = False

universal = {
    'exp_name': exp_name,
    'fname_dataset': fname_dataset,
    'savepath': path_output,
    'indexpath': os.path.join(path_data, 'TraValIndex_M=50000.pkl'),
    'configpath': path_config
}
device_no = 0
#device = utils.set_device(device_no)
transform = 'sample_norm_cent'

parameters = {
    'model': 'AEC',
    'mode': 'train',
    'n_epochs': 20,
    'show': False,
    'send_message': False,
    'early_stopping': True,
    'patience': 10,
    'transform': transform,
    'img_index': str(img_index)[1:-1],
    'tb': True,
    'tbport': 6999,
    'workers': 5,
    'loadmode': 'ram',
    'datafiletype': 'h5'
}
hyperparameters = {
    'batch_size': '5',
    'lr': '0.0001'
}
init_path = utils.config_training(universal, parameters, hyperparameters)
config_AEC = utils.Configuration(init_path)
config_AEC.load_config()


config_AEC.init_exp_env()

config_AEC.set_device(device_no)
config_AEC.show = True

#config = utils.Configuration(init_path)

print(os.path.abspath(init_path))
#print(init_path)
#/home/julia/Cluster/Config/init_train.ini

train(config_AEC)