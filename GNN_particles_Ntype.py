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
# matplotlib.use("Qt5Agg")
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torchvision.transforms import GaussianBlur
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.data_loaders import *
from ParticleGraph.embedding_cluster import *

from ParticleGraph.generators.utils import *
from ParticleGraph.generators.graph_data_generator import *
from ParticleGraph.models.graph_trainer import *

from ParticleGraph.models import Siren_Network
from ParticleGraph.models.Ghost_Particles import Ghost_Particles
from ParticleGraph.models.utils import *
from ParticleGraph.utils import *

if __name__ == '__main__':

    config_list =['boids_16_dropout_10', 'boids_16_dropout_10_no_ghost']

    seed_list = np.arange(10)

    for config_file in config_list:
        # for seed in seed_list:
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        # config.dataset = f'{config.dataset}_{seed}'

        device = set_device(config.training.device)
        print(f'device {device}')

        data_generate(config, device=device, visualize=True, run_vizualized=0, style='frame color', alpha=1, erase=True, bSave=True, step=10) #config.simulation.n_frames // 2)
        data_train(config, config_file, device)
        # data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False, best_model=20, run=0, step=83, test_simulation=False, sample_embedding=False, device=device)    # config.simulation.n_frames // 7

