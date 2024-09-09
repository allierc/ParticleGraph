import time
from shutil import copyfile

import networkx as nx
import scipy.io
import torch
# import networkx as nx
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

if __name__ == '__main__':

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    # config_list = ["arbitrary_3_cell_sequence_d_bis"]
    # config_list = ["arbitrary_3_cell_sequence_d_a"]
    # config_list = ["arbitrary_3_cell_sequence_f"]
    # config_list = ["arbitrary_3_cell_sequence_d_3"]

    # config_list = ["arbitrary_3_cell"]
    # config_list = ["arbitrary_3_cell_tracking"]
    # config_list = ["boids_division_tracking_A"]
    # config_list = ["boids_division_tracking_B"]
    # config_list = ["boids_division_tracking_ctrl"]
    # config_list = ['boids_division_model_2_tracking']

    # config_list = ["boids_division_tracking_A", "boids_division_tracking_B"]

    # config_list = ["gland_SMG2-processed"]
    # config_list = ["MDCK_FN_PB-Kleb_JF585"]

    # config_list = ['signal_N2_b_r1']
    # config_list = ['signal_N2_b_r2']
    # config_list = ['signal_N2_b_r5']
    config_list = ['signal_N2_hemibrain_3_r1_a']

    # config_list = ['gravity_solar_system_d']

    for config_file in config_list:

        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        # data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=False, bSave=True, step=10) # config.simulation.n_frames // config.simulation.n_frames)
        data_train(config, config_file, True, device)
        # data_test (config=config, config_file=config_file, visualize=True, style='color', verbose=False, best_model='1', run=0, step=25, test_simulation=True, sample_embedding=False, device=device)    # config.simulation.n_frames // 7
