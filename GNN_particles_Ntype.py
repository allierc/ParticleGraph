import time
from shutil import copyfile
import argparse
import networkx as nx
import scipy.io
import umap
import torch
import torch.nn as nn
import torch_geometric.data as data
from sklearn import metrics
from tifffile import imread
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torchvision.transforms import GaussianBlur
from matplotlib import pyplot as plt

from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.utils import *
from ParticleGraph.generators.graph_data_generator import *
from ParticleGraph.models.graph_trainer import *
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *
import warnings

if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    parser = argparse.ArgumentParser(description="ParticleGraph")
    parser.add_argument('-o', '--option', nargs='+', help='Option that takes multiple values')

    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option!=None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:

        task = 'test'
        best_model = ''
        # config_list = ['fluids_m17_1']
        config_list = ['multimaterial_9_9','multimaterial_2', 'multimaterial_8_1', 'multimaterial_8_8', 'multimaterial_8_11', 'multimaterial_8_12',
                    'multimaterial_9_7', 'multimaterial_9_8','multimaterial_9_10', 'multimaterial_9_11', 'multimaterial_9_12', 'multimaterial_9_13', 'multimaterial_9_14',
                    'multimaterial_10_1','multimaterial_10_2', 'multimaterial_10_3']
        # config_list = ['signal_N2_a38_1']
        # config_list = ['gravity_16']
        # config_list = ['cell_MDCK_12']
        # config_list = ['signal_N2_a43_17']
        # config_list = ['arbitrary_3_5','arbitrary_3_7','arbitrary_3_8','arbitrary_3_9']
        # config_list = ['gravity_16_1', 'gravity_16_2']

    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f'config_file  {config.config_file}')
        print(f'device  {device}')
        print(f'folder  {config.dataset}')

        if 'generate' in task:
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='black color', alpha=1, erase=False, bSave=True, step=100)  #config.simulation.n_frames // 100)
        if 'train' in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)
        if 'test' in task:
            data_test(config=config, visualize=True, style='black color', verbose=False, best_model='best', run=17, test_mode='fixed_bounce_all', sample_embedding=False, step=4, device=device) # particle_of_interest=100,
        if 'try_func' in task:
            try_func(max_radius=config.simulation.max_radius, device=device)




# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"

