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

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

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
        task = 'generate_train'
        best_model = None
        # config_list = ['fluids_m13_1_training']
        # config_list = ['falling_water_ramp_x6_12']
        config_list =['signal_N2_a_SNR7_ter']

    for config_file in config_list:

        print('')
        print(f'config_file {config_file}')
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        if 'generate' in task:
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='black field', alpha=1, erase=False, bSave=True, step=4)  #config.simulation.n_frames // 100)
        if 'train' in task:
            data_train(config=config, config_file=config_file, erase=False, best_model=best_model, device=device)
        if 'test' in task:
            # data_test(config=config, config_file=config_file, visualize=True, style='white bw color', verbose=False, best_model='best', run=0, plot_data=False,
            #           test_simulation=False, sample_embedding=False, fixed=False, bounce=False, step=4, device=device)


            data_test(config=config, config_file=config_file, visualize=True, style='black color', verbose=False, best_model='best', run=1, plot_data=True,
                      test_simulation=False, sample_embedding=False, fixed=True, bounce=True, step=4, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style='black color', verbose=False, best_model='best', run=2, plot_data=False,
                      test_simulation=False, sample_embedding=False, fixed=True, bounce=False, step=4, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style='black color', verbose=False, best_model='best', run=15, plot_data=False,
                      test_simulation=False, sample_embedding=False, fixed=True, bounce=True, step=4, device=device)


# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"

