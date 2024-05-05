import glob
import logging
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import torch
# import networkx as nx
import torch.nn as nn
import torch_geometric.data as data
import umap
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
import os
import scipy.io
from sklearn import metrics
from matplotlib import rc
import matplotlib
from tifffile import imread
import networkx as nx
# matplotlib.use("Qt5Agg")
from torchvision.transforms import GaussianBlur

from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import imageio

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.generator_initialization import init_particles, init_mesh
from ParticleGraph.generators.utils import choose_model, choose_mesh_model, generate_from_data
from ParticleGraph.models.utils import *
from ParticleGraph.models.Ghost_Particles import Ghost_Particles

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

from ParticleGraph.data_loaders import *
from ParticleGraph.utils import *
from ParticleGraph.fitting_models import linear_model
from ParticleGraph.embedding_cluster import *
from ParticleGraph.models import Division_Predictor
# from ParticleGraph.Plot3D import *
from ParticleGraph.generators import PDE_B
from ParticleGraph.models import Siren_Network



def data_generate(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):
    print('')

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    has_signal = (config.graph_model.signal_model_name != '')
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_cell_division = simulation_config.has_cell_division
    n_frames = simulation_config.n_frames
    cycle_length = None
    has_particle_dropout = training_config.particle_dropout > 0
    noise_level = training_config.noise_level
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    if config.data_folder_name != 'none':
        generate_from_data(config=config, device=device, visualize=visualize, folder=folder, step=step)
        return

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/generated_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/generated_data/*')
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    model, bc_pos, bc_dpos = choose_model(config, device=device)
    if has_mesh:
        mesh_model = choose_mesh_model(config, device=device)
    else:
        mesh_model = None
    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_particles)
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1, cycle_length, cycle_length_distrib = init_particles(config, device=device,
                                                                                    cycle_length=cycle_length)

        # create differnet initial conditions
        # X1[:, 0] = X1[:, 0] / n_particle_types - torch.ones_like(X1[:,0])*0.5
        # for n in range(n_particle_types):
        #     X1[index_particles[n], 0] = X1[index_particles[n], 0] + n / n_particle_types

        if has_mesh:
            X1_mesh, V1_mesh, T1_mesh, H1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)
            torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
            mask_mesh = mesh_data['mask'].squeeze()
        if only_mesh | (model_config.particle_model_name == 'PDE_O'):
            X1 = X1_mesh.clone().detach()
            H1 = H1_mesh.clone().detach()
            T1 = T1_mesh.clone().detach()
        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1 == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            # calculate cell division
            if (it >= 0) & has_cell_division & (n_particles < 20000):
                pos = torch.argwhere(A1.squeeze() > cycle_length_distrib)
                y_division = (A1.squeeze() > cycle_length_distrib).clone().detach() * 1.0
                # cell division
                if len(pos) > 1:
                    n_add_nodes = len(pos)
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)

                    y_division = torch.concatenate((y_division, torch.zeros((n_add_nodes), device=device)), 0)

                    n_particles = n_particles + n_add_nodes
                    N1 = torch.arange(n_particles, device=device)
                    N1 = N1[:, None]
                    separation = 1E-3 * torch.randn((n_add_nodes, 2), device=device)
                    X1 = torch.cat((X1, X1[pos, :] + separation), dim=0)
                    X1[pos, :] = X1[pos, :] - separation
                    phi = torch.randn(n_add_nodes, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)
                    new_x = cos_phi * V1[pos, 0] + sin_phi * V1[pos, 1]
                    new_y = -sin_phi * V1[pos, 0] + cos_phi * V1[pos, 1]
                    V1[pos, 0] = new_x
                    V1[pos, 1] = new_y
                    V1 = torch.cat((V1, -V1[pos, :]), dim=0)
                    T1 = torch.cat((T1, T1[pos, :]), dim=0)
                    H1 = torch.cat((H1, H1[pos, :]), dim=0)
                    A1[pos, :] = 0
                    A1 = torch.cat((A1, A1[pos, :]), dim=0)
                    nd = torch.ones(len(pos), device=device) + 0.05 * torch.randn(len(pos), device=device)
                    cycle_length_distrib = torch.cat(
                        (cycle_length_distrib, cycle_length[to_numpy(T1[pos, 0])].squeeze() * nd), dim=0)
                    y_timer = A1.squeeze().clone().detach()

                    index_particles = []
                    for n in range(n_particles):
                        pos = torch.argwhere(T1 == n)
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)
            if has_mesh:
                x_mesh = torch.concatenate(
                    (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                     T1_mesh.clone().detach(), H1_mesh.clone().detach()), 1)
                dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'],
                                         edge_attr=mesh_data['edge_weight'], device=device)

            if not(only_mesh):
                # compute connectivity rule
                if has_adjacency_matrix:
                    adj_t = adjacency > 0
                    edge_index = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
                else:
                    distance = torch.sum(bc_dpos(x[:, None, 1:dimension+1] - x[None, :, 1:dimension+1]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                # model prediction
                with torch.no_grad():
                    y = model(dataset)

                # append list
                if (it >= 0) & bSave:

                    if has_cell_division:
                        x_list.append(x.clone().detach())
                        y_ = torch.concatenate((y, y_timer[:, None], y_division[:, None]), 1)
                        y_list.append(y_.clone().detach())
                    else:
                        if has_particle_dropout:
                            x_ = x[particle_dropout_mask].clone().detach()
                            x_[:, 0] = torch.arange(len(x_), device=device)
                            x_list.append(x_)
                            x_ = x[inv_particle_dropout_mask].clone().detach()
                            x_[:, 0] = torch.arange(len(x_), device=device)
                            x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                            y_list.append(y[particle_dropout_mask].clone().detach())
                        else:
                            x_list.append(x.clone().detach())
                            y_list.append(y.clone().detach())
                            if (run == 1) & (it == 200):
                                torch.save(x, f'graphs_data/graphs_{dataset_name}/x_200.pt')
                                torch.save(y, f'graphs_data/graphs_{dataset_name}/y_200.pt')

                # Particle update
                if model_config.particle_model_name == 'PDE_O':
                    H1[:, 2] = H1[:, 2] + y.squeeze() * delta_t
                    X1[:, 0] = H1[:, 0] + (3 / 8) * mesh_data['size'] * torch.cos(H1[:, 2])
                    X1[:, 1] = H1[:, 1] + (3 / 8) * mesh_data['size'] * torch.sin(H1[:, 2])
                    X1 = bc_pos(X1)
                if has_signal:
                    H1[:, 1] = y.squeeze()
                    H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t
                else:
                    if model_config.prediction == '2nd_derivative':
                        V1 += y * delta_t
                    else:
                        V1 = y
                    X1 = bc_pos(X1 + V1 * delta_t)
                    if config.graph_model.mesh_model_name == 'Chemotaxism_Mesh':
                        grad = grads2D(torch.reshape(H1_mesh[:, 0], (300, 300)))
                        x_ = np.clip(to_numpy(X1[:, 0]) * 300, 0, 299)
                        y_ = np.clip(to_numpy(X1[:, 1]) * 300, 0, 299)
                        X1[:, 0] += torch.clamp(grad[1][y_, x_] / 5E4, min=-0.5, max=0.5)
                        X1[:, 1] += torch.clamp(grad[0][y_, x_] / 5E4, min=-0.5, max=0.5)

                A1 = A1 + delta_t

            # Mesh update
            if has_mesh:
                x_mesh_list.append(x_mesh.clone().detach())
                match config.graph_model.mesh_model_name:
                    case 'DiffMesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                            H1[mask_mesh, 1:2] = pred[mask_mesh]
                        H1_mesh[mask_mesh, 0:1] += pred[mask_mesh, 0:1] * delta_t
                        new_pred = torch.zeros_like(pred)
                        new_pred[mask_mesh] = pred[mask_mesh]
                        pred = new_pred
                    case 'WaveMesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                        H1_mesh[mask_mesh, 1:2] += pred[mask_mesh, :] * delta_t
                        H1_mesh[mask_mesh, 0:1] += H1_mesh[mask_mesh, 1:2] * delta_t
                        # x_ = to_numpy(x_mesh)
                        # plt.scatter(x_[:, 1], x_[:, 2], c=to_numpy(H1_mesh[:, 0]))
                    case 'RD_Gray_Scott_Mesh' | 'RD_FitzHugh_Nagumo_Mesh' | 'RD_RPS_Mesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                            H1_mesh[mesh_data['mask'].squeeze(), :] += pred[mesh_data['mask'].squeeze(), :] * delta_t
                            H1 = H1_mesh.clone().detach()
                    case 'Chemotaxism_Mesh':
                        with torch.no_grad():
                            pred = mesh_model(dataset_mesh)
                            H1_mesh[mesh_data['mask'].squeeze(), :] += pred[mesh_data['mask'].squeeze(), :] * delta_t
                            distance = torch.sum(bc_dpos(x[:, None, 1:3] - x_mesh[None, :, 1:3]) ** 2, dim=2)
                            distance = distance < 0.00015
                            distance = torch.sum(distance, dim=0)
                            H1_mesh = torch.relu(H1_mesh * 1.0012 - 2 * distance[:, None])
                            H1_mesh = torch.clamp(H1_mesh, min=0, max=5000)
                    case 'PDE_O_Mesh':
                        pred = []

                y_mesh_list.append(pred)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'graph' in style:

                    fig = plt.figure(figsize=(10, 10))
                    # distance2 = torch.sum((x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
                    # adj_t2 = ((distance2 < max_radius ** 2) & (distance2 < 0.9 ** 2)).float() * 1
                    # edge_index2 = adj_t2.nonzero().t().contiguous()
                    # dataset2 = data.Data(x=x, edge_index=edge_index2)
                    # pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                    # vis = to_networkx(dataset2, remove_self_loops=True, to_undirected=True)
                    # nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=alpha)

                    if model_config.mesh_model_name == 'RD_RPS_Mesh':
                        H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                        plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                    elif (model_config.mesh_model_name == 'Wave_Mesh') | (model_config.mesh_model_name =='DiffMesh') :
                        pts = x_mesh[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                        if model_config.mesh_model_name == 'WaveMesh' :
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=5000)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    elif model_config.particle_model_name == 'PDE_G':
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=40, color=cmap.color(n))
                    elif model_config.particle_model_name == 'PDE_E':
                        for n in range(n_particle_types):
                            g = 40
                            if simulation_config.params[n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n),
                                        alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'bw' in style:

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    s_p = 50
                    if simulation_config.has_cell_division:
                        s_p = 25
                    if False:  # config.simulation.non_discrete_level>0:
                        plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=s_p, color='k')
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                        s=s_p, color='k')
                    if training_config.particle_dropout > 0:
                        plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                    x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                    alpha=0.75)
                        plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                 x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                if 'color' in style:

                    if model_config.particle_model_name == 'PDE_O':
                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=100,
                                    c=np.sin(to_numpy(H1[:, 2])), vmin=-1, vmax=1, cmap='viridis')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Lut_Fig_{run}_{it}.jpg",
                                    dpi=170.7)
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        # plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=5, c='b')
                        plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c='lawngreen',
                                    alpha=0.75)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Rot_{run}_Fig{it}.jpg",
                                    dpi=170.7)
                        plt.close()

                    elif model_config.signal_model_name == 'PDE_N':

                        matplotlib.rcParams['savefig.pad_inches'] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = plt.axes([0, 0, 1, 1], frameon=False)
                        pos = dict(enumerate(np.array(x[:, 1:3].detach().cpu()), 0))
                        vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
                        nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, alpha=0.025)
                        plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=100, c=to_numpy(H1[:, 0]), cmap='viridis')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        plt.xlim([-1.5, 1.5])
                        plt.ylim([-1.5, 1.5])
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                        plt.close()

                    elif model_config.mesh_model_name == 'Chemotaxism_Mesh':

                        # dx_ = to_numpy(grad[1][y_,x_])/100
                        # dy_ = to_numpy(grad[0][y_,x_])/100
                        # H1_IM = torch.reshape(H1_mesh[:, 0], (300, 300))
                        # plt.imshow(to_numpy(grad[1]+grad[0]), cmap='viridis')
                        # for i in range(1700):
                        #     plt.arrow(x=x_[i],y=y_[i],dx=dx_[i],dy=dy_[i], head_width=2, length_includes_head=True, color='w')

                        fig = plt.figure(figsize=(12, 12))
                        H1_IM = torch.reshape(H1_mesh[:, 0], (300, 300))
                        plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=5000, cmap='viridis')
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy() * 300,
                                        x[index_particles[n], 2].detach().cpu().numpy() * 300, s=1, color='w')
                        plt.xlim([0, 300])
                        plt.ylim([0, 300])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/All__{run}_{it}.jpg", dpi=170.7)
                        plt.close()

                        # fig = plt.figure(figsize=(12,12))
                        # H1_IM = torch.reshape(distance, (300, 300))
                        # plt.imshow(H1_IM.detach().cpu().numpy()*30, vmin=0, vmax=500)
                        # for n in range(n_particle_types):
                        #     plt.scatter(x[index_particles[n], 1].detach().cpu().numpy() * 300,
                        #                 x[index_particles[n], 2].detach().cpu().numpy() * 300, s=1, color='w')
                        # plt.xlim([0, 300])
                        # plt.ylim([0, 300])
                        # plt.xticks([])
                        # plt.yticks([])
                        # plt.tight_layout()
                        # plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Boids_{it}.jpg", dpi=170.7)
                        # plt.close()

                    elif (model_config.particle_model_name == 'PDE_A') & (dimension == 3):

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111, projection='3d')
                        for n in range(n_particle_types):
                            ax.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 3]), s=50, color=cmap.color(n))
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.set_zlim([0, 1])
                        pl.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                        plt.close()

                    else:
                        # matplotlib.use("Qt5Agg")

                        plt.rcParams['text.usetex'] = True
                        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                        matplotlib.rcParams['savefig.pad_inches'] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        # ax.xaxis.get_major_formatter()._usetex = False
                        # ax.yaxis.get_major_formatter()._usetex = False
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        # if (has_mesh | (simulation_config.boundary == 'periodic')):
                        #     ax = plt.axes([0, 0, 1, 1], frameon=False)
                        # else:
                        #     ax = plt.axes([-2, -2, 2, 2], frameon=False)
                        # ax.get_xaxis().set_visible(False)
                        # ax.get_yaxis().set_visible(False)
                        # plt.autoscale(tight=True)
                        if has_mesh:
                            pts = x_mesh[:, 1:3].detach().cpu().numpy()
                            tri = Delaunay(pts)
                            colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                            match model_config.mesh_model_name:
                                case 'DiffMesh':
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                                case 'WaveMesh':
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                                case 'RD_Gray_Scott_Mesh':
                                    fig = plt.figure(figsize=(12, 6))
                                    ax = fig.add_subplot(1, 2, 1)
                                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.axis('off')
                                    ax = fig.add_subplot(1, 2, 2)
                                    colors = torch.sum(x[tri.simplices, 7], dim=1) / 3.0
                                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.axis('off')
                                case 'RD_RPS_Mesh':
                                    fig = plt.figure(figsize=(12, 12))
                                    H1_IM = torch.reshape(H1, (100, 100, 3))
                                    plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.axis('off')
                        else:
                            s_p = 100
                            if simulation_config.has_cell_division:
                                s_p = 25
                            if False:  # config.simulation.non_discrete_level>0:
                                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=s_p, color='k')
                            else:
                                for n in range(n_particle_types):
                                    plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                                s=s_p, color=cmap.color(n))
                            if training_config.particle_dropout > 0:
                                plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                            alpha=0.75)
                                plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                         x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                        plt.xlim([0,1])
                        plt.ylim([0,1])
                        if 'frame' in style:
                            plt.xlabel(r'$x$', fontsize=64)
                            plt.ylabel(r'$y$', fontsize=64)
                            plt.xticks(fontsize=32.0)
                            plt.yticks(fontsize=32.0)
                        else:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{run}_{it}.jpg", dpi=170.7)
                        plt.close()

                        if False:  # not(has_mesh):
                            fig = plt.figure(figsize=(12, 12))
                            s_p = 25
                            if simulation_config.has_cell_division:
                                s_p = 10
                            for n in range(n_particle_types):
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color='k')
                            if (simulation_config.boundary == 'periodic'):
                                plt.xlim([0, 1])
                                plt.ylim([0, 1])
                            else:
                                plt.xlim([-4, 4])
                                plt.ylim([-4, 4])
                            plt.xticks([])
                            plt.yticks([])
                            plt.tight_layout()
                            plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_bw_{it}.jpg", dpi=170.7)
                            plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')
            torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
            torch.save(cycle_length_distrib, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')


def data_generate_particle_field(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):
    print('')

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    has_signal = (config.graph_model.signal_model_name != '')
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = True
    only_mesh = False
    has_cell_division = False

    n_nodes = simulation_config.n_nodes
    n_frames = simulation_config.n_frames
    cycle_length = None
    has_particle_dropout = training_config.particle_dropout > 0
    noise_level = training_config.noise_level
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    if config.data_folder_name != 'none':
        generate_from_data(config=config, device=device, visualize=visualize, folder=folder, step=step)
        return

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/generated_data/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/generated_data/*')
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    model, bc_pos, bc_dpos = choose_model(config, device=device)

    params = config.simulation.params
    p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
    for n in range(n_particle_types):
        p[n] = torch.tensor(params[n][0:3], device=device)

    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_particles)
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1, cycle_length, cycle_length_distrib = init_particles(config, device=device, cycle_length=cycle_length)
        X1_mesh, V1_mesh, T1_mesh, H1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)
        T1_mesh = -1.0 + T1_mesh * -1.0
        A1_mesh = torch.zeros_like(T1_mesh)
        torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
        mask_mesh = mesh_data['mask'].squeeze()

        a1 = 1E-2  # diffusion coefficient
        a2 = 8E-5  # positive rate coefficient
        a3 = 6.65E-5  # negative rate coefficient

        i0 = imread(f'graphs_data/{config.simulation.node_diffusion_map}')
        index = np.round(i0[(to_numpy(X1_mesh[:, 0]) * 255).astype(int), (to_numpy(X1_mesh[:, 1]) * 255).astype(int)] ).astype(int)
        coeff_diff = a1 * np.array(config.simulation.diffusion_coefficients)[index]
        model.coeff_diff = torch.tensor(coeff_diff, device=device)
        i0 = imread(f'graphs_data/{config.simulation.node_proliferation_map}')
        index = np.round(i0[(to_numpy(X1_mesh[:, 0]) * 255).astype(int), (to_numpy(X1_mesh[:, 1]) * 255).astype(int)]).astype(int)
        pos_rate = a2 * np.array(config.simulation.pos_rate)[index]
        model.pos_rate = torch.tensor(pos_rate, device=device)
        model.neg_rate = - torch.ones_like(model.pos_rate) * a3 * torch.tensor(config.simulation.pos_rate[0], device=device)

        # matplotlib.use("Qt5Agg")
        # fig = plt.figure(figsize=(12, 12))
        # plt.imshow(i0)
        # plt.imshow(np.reshape(proliferation_rate, (100, 100)), cmap='viridis')
        # plt.colorbar()


        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1 == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            # calculate cell division
            if (it >= 0) & has_cell_division & (n_particles < 20000):
                pos = torch.argwhere(A1.squeeze() > cycle_length_distrib)
                y_division = (A1.squeeze() > cycle_length_distrib).clone().detach() * 1.0
                # cell division
                if len(pos) > 1:
                    n_add_nodes = len(pos)
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)

                    y_division = torch.concatenate((y_division, torch.zeros((n_add_nodes), device=device)), 0)

                    n_particles = n_particles + n_add_nodes
                    N1 = torch.arange(n_particles, device=device)
                    N1 = N1[:, None]
                    separation = 1E-3 * torch.randn((n_add_nodes, 2), device=device)
                    X1 = torch.cat((X1, X1[pos, :] + separation), dim=0)
                    X1[pos, :] = X1[pos, :] - separation
                    phi = torch.randn(n_add_nodes, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)
                    new_x = cos_phi * V1[pos, 0] + sin_phi * V1[pos, 1]
                    new_y = -sin_phi * V1[pos, 0] + cos_phi * V1[pos, 1]
                    V1[pos, 0] = new_x
                    V1[pos, 1] = new_y
                    V1 = torch.cat((V1, -V1[pos, :]), dim=0)
                    T1 = torch.cat((T1, T1[pos, :]), dim=0)
                    H1 = torch.cat((H1, H1[pos, :]), dim=0)
                    A1[pos, :] = 0
                    A1 = torch.cat((A1, A1[pos, :]), dim=0)
                    nd = torch.ones(len(pos), device=device) + 0.05 * torch.randn(len(pos), device=device)
                    cycle_length_distrib = torch.cat(
                        (cycle_length_distrib, cycle_length[to_numpy(T1[pos, 0])].squeeze() * nd), dim=0)
                    y_timer = A1.squeeze().clone().detach()

                    index_particles = []
                    for n in range(n_particles):
                        pos = torch.argwhere(T1 == n)
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)

            x_mesh = torch.concatenate((N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                 T1_mesh.clone().detach(), H1_mesh.clone().detach(), A1_mesh.clone().detach()), 1)

            x_particle_field = torch.concatenate((x_mesh, x), 0)
            x_particle_field [:,0] = torch.arange(x_mesh.shape[0]+x.shape[0], device=device)

            # compute connectivity rule
            distance = torch.sum(bc_dpos(x_particle_field[:, None, 1:dimension+1] - x_particle_field[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_particle = adj_t.nonzero().t().contiguous()

            dataset = data.Data(x=x_particle_field, pos=x[:, 1:3], edge_index=edge_index, edge_particle=edge_particle, edge_mesh=mesh_data['edge_index'], edge_attr=mesh_data['edge_weight'])

            # model prediction
            with torch.no_grad():
                y , drift, y_mesh  = model(dataset)

            # append list
            if (it >= 0) & bSave:

                if has_cell_division:
                    x_list.append(x.clone().detach())
                    y_ = torch.concatenate((y, y_timer[:, None], y_division[:, None]), 1)
                    y_list.append(y_.clone().detach())
                else:
                    if has_particle_dropout:
                        x_ = x[particle_dropout_mask].clone().detach()
                        x_[:, 0] = torch.arange(len(x_), device=device)
                        x_list.append(x_)
                        x_ = x[inv_particle_dropout_mask].clone().detach()
                        x_[:, 0] = torch.arange(len(x_), device=device)
                        x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                        y_list.append(y[particle_dropout_mask].clone().detach())
                    else:
                        x_list.append(x.clone().detach())
                        y_list.append(y.clone().detach())

            # Particle update

            V1 += (y + drift) * delta_t
            X1 = bc_pos(X1 + V1 * delta_t)

            A1 = A1 + delta_t

            # Mesh update

            x_mesh_list.append(x_mesh.clone().detach())

            # H1_mesh[mask_mesh, 0:1] += y_mesh[mask_mesh] * delta_t
            # new_pred = torch.zeros_like(y_mesh)
            # new_pred[mask_mesh] = y_mesh[mask_mesh]
            # y_mesh_list.append(new_pred)

            H1_mesh[:, 0:1] += y_mesh * delta_t
            H1_mesh[:, 0:1] = torch.clamp(H1_mesh[:, 0:1], min=0, max=7500)
            y_mesh_list.append(y_mesh)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                fig = plt.figure(figsize=(10, 10))
                pts = x_mesh[:, 1:3].detach().cpu().numpy()
                tri = Delaunay(pts)
                colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                if model_config.mesh_model_name == 'WaveMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                  vmax=2500)
                else:
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=5000)
                plt.xlim([0, 1])
                plt.ylim([0, 1])




                # H1_IM = torch.reshape(H1_mesh[:, 0], (100, 100))
                # plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=5000, cmap='viridis')
                for n in range(n_particle_types):
                    plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                x[index_particles[n], 2].detach().cpu().numpy(), s=0.5, color='w')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/All_{run}_{it}.jpg", dpi=170.7)
                plt.close()


        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')
            torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
            torch.save(cycle_length_distrib, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')



def data_train(config, device):

    has_mesh = (config.graph_model.mesh_model_name != '')
    has_signal = (config.graph_model.signal_model_name != '')
    has_particle_field = (config.graph_model.particle_model_name == 'PDE_ParticleField')

    if has_particle_field:
        data_train_particle_field(config, device)
    elif has_mesh:
        data_train_mesh(config, device)
    elif has_signal:
        data_train_signal(config, device)
    else:
        data_train_particles(config, device)



def data_train_particles(config, device):
    print('')

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    n_ghosts = train_config.n_ghosts
    has_large_range = train_config.large_range
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)

    l_dir, log_dir, logger = create_log_dir(config, dataset_name)
    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs}')
    logger.info(f'Graph files N: {NGraphs}')
    time.sleep(0.5)

    x_list = []
    y_list = []
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in trange(n_frames):
            if (k % 10 == 0) | (n_frames < 1000):
                x = torch.cat((x, x_list[run][k].clone().detach()), 0)
                y = torch.cat((y, y_list[run][k].clone().detach()), 0)
        print(x_list[run][k].shape)
        time.sleep(0.5)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_20.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()
    if has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()
    if dimension == 2:
        type_list = x[:, 5:6].clone().detach()
    elif dimension == 3:
        type_list = x[:, 7:8].clone().detach()
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    if has_ghost:

        if model_config.particle_model_name == 'B':
            print('Train SIREN model ...')

            run = 1 + np.random.randint(NGraphs - 1)
            frame = 1 + np.random.randint(n_frames - 2)

            image_width = 256
            model_input = get_mgrid(image_width, 2)
            model_input = model_input.cuda()

            for frame in range(0,n_frames,50):


                model_siren = Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=256, hidden_layers=8, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.)
                x = x_list[run][frame].clone().detach()

                pos_coords = x[:, 1:3].clone().detach() * image_width
                t_coords = k / n_frames * torch.ones_like(pos_coords[:, 0:1], device=device)
                coords = pos_coords  # torch.cat((pos_coords, t_coords), 1)
                y = x[:, 3:5].clone().detach() / vnorm

                # image_file = './graphs_data/pattern_1.tif'  # boat_512.tif, beads_abb.tif, beads_gt.tif
                # target = load_image(image_file, crop_width=image_width, device=device)

                target = torch.zeros((2, image_width, image_width), device=device)
                target[0,pos_coords[:, 0].long(), pos_coords[:, 1].long()] = -y[:,0].squeeze()
                target[0:1,:,:] = GaussianBlur(51,10)(target[0:1,:,:])
                target[1,pos_coords[:, 0].long(), pos_coords[:, 1].long()] = y[:,1].squeeze()
                target[1:2,:,:] = GaussianBlur(51,10)(target[1:2,:,:])

                # matplotlib.use("Qt5Agg")
                # fig = plt.figure(figsize=(8, 8))
                # plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=20, c=to_numpy(y[:,0].squeeze()))
                # fig = plt.figure(figsize=(8, 8))
                # plt.imshow(to_numpy(target[0, :, :].squeeze()))
                # plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=20, c=to_numpy(y[:, 0].squeeze()))

                total_steps = 1000
                steps_til_summary = 200
                optim = torch.optim.Adam(lr=1e-4, params=model_siren.parameters())

                for step in trange(total_steps + 1):
                    model_output, coords = model_siren()
                    img_grad_ = gradient(model_output, coords)
                    # loss = ((model_output - ground_truth) ** 2).mean()
                    loss = (img_grad_[:,1].view(image_width, image_width) - target[0]).norm(2) + (img_grad_[:,0].view(image_width, image_width) - target[1]).norm(2)

                    # if not step % steps_til_summary:
                    #     print("Step %d, Total loss %0.6f" % (step, loss))

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # matplotlib.use("Qt5Agg")

                fig = plt.figure(figsize=(16, 8))
                ax = fig.add_subplot(2, 4, 2)
                plt.imshow(to_numpy(target[0, :, :].squeeze()),vmin=-0.1,vmax=0.1)
                plt.title('Velocity_field_y')
                plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
                plt.xlim([0, image_width])
                plt.ylim([0, image_width])
                plt.xticks([])
                plt.yticks([])
                ax.invert_yaxis()
                ax = fig.add_subplot(2, 4, 1)
                plt.imshow(to_numpy(target[1, :, :].squeeze()),vmin=-0.1,vmax=0.1)
                plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
                plt.xlim([0, image_width])
                plt.ylim([0, image_width])
                plt.xticks([])
                plt.yticks([])
                ax.invert_yaxis()
                plt.title('Velocity_field_x')
                ax = fig.add_subplot(1, 2, 2)
                temp = -model_output.cpu().view(image_width,image_width).permute(1,0).detach().numpy()
                plt.imshow(temp)
                plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=1, color='w')
                plt.xlim([0, image_width])
                plt.ylim([0, image_width])
                plt.xticks([])
                plt.yticks([])
                ax.invert_yaxis()
                plt.title('Reconstructed potential')
                ax = fig.add_subplot(2, 4, 6)
                plt.imshow(img_grad_[:, 1].cpu().view(image_width, image_width).detach().numpy(),vmin=-0.1,vmax=0.1)
                plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
                plt.xlim([0, image_width])
                plt.ylim([0, image_width])
                plt.xticks([])
                plt.yticks([])
                ax.invert_yaxis()
                plt.title('Gradient_y from potential')
                ax = fig.add_subplot(2, 4, 5)
                plt.imshow(img_grad_[:, 0].cpu().view(image_width, image_width).detach().numpy(),vmin=-0.1,vmax=0.1)
                plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
                plt.xlim([0, image_width])
                plt.ylim([0, image_width])
                plt.xticks([])
                plt.yticks([])
                ax.invert_yaxis()
                plt.title('Gradient_x from potential')
                plt.tight_layout()
                plt.savefig(f"{log_dir}/tmp_training/siren/siren_{frame}.jpg", dpi=170.7)
                plt.close()

            torch.save({'model_state_dict': model_siren.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', f'Siren_model'))

        ghosts_particles = Ghost_Particles(config, n_particles, vnorm, device)
        optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if (epoch == 1) & (has_ghost):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        total_loss_division = 0
        Niter = n_frames * data_augmentation_loop // batch_size

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            time_batch = []

            for batch in range(batch_size):

                k = 1 + np.random.randint(n_frames - 2)

                x = x_list[run][k].clone().detach()

                if has_ghost:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance,axis=1)[1]
                        x_ghost[:,3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)



                    with torch.no_grad():
                        model.a[run,n_particles:n_particles+n_ghosts] = model.a[run,ghosts_particles.embedding_index].clone().detach()   # sample ghost embedding

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                t = torch.Tensor([max_radius ** 2])
                edges = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x[:, :], edge_index=edges)
                dataset_batch.append(dataset)

                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)

                y = y / ynorm

                if data_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y
                if batch == 0:
                    y_batch = y[:, 0:2]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:2]), dim=0)

                if has_cell_division:
                    if batch == 0:
                        time_batch = torch.concatenate((x[:, 0:1], torch.ones_like(y[:, 3:4], device=device) * k),
                                                       dim=1)
                        y_batch_division = y[:, 2:3]
                    else:
                        time_batch = torch.concatenate((time_batch, torch.concatenate(
                            (x[:, 0:1], torch.ones_like(y[:, 2:3], device=device) * k), dim=1)), dim=0)
                        y_batch_division = torch.concatenate((y_batch_division, y[:, 3:4]), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()
            if has_cell_division:
                optimizer_division.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi)

            if has_cell_division:
                pred_division = model_division(time_batch, data_id=run)
                loss_division = (pred_division - y_batch_division).norm(2)
                loss_division.backward()
                optimizer_division.step()
                total_loss_division += loss_division.item()

            if has_ghost:
                # if simulation_config.boundary == 'no':
                #     current_distribution = torch.std(x[:, 1:3], dim=0).clone().detach()
                #     loss = ((pred[mask_ghost] - y_batch)).norm(2) + 1E2 * (torch.std(x_ghost[:, 1:3]) - current_distribution).norm(2)  # loss_ghost_distribution
                loss = ((pred[mask_ghost] - y_batch)).norm(2)

            else:
                if not (has_large_range):
                    loss = (pred - y_batch).norm(2)
                else:
                    loss = ((pred - y_batch) / (y_batch)).norm(2) / 1E9

            visualize_embedding = True
            if visualize_embedding & (((epoch == 0) & (N < 10000) & (N % 200 == 0)) | (N==0)):
                plot_training(config=config, dataset_name=dataset_name, model_name=model_config.particle_model_name, log_dir=log_dir,
                              epoch=epoch, N=N, x=x, model=model, n_nodes=0, n_node_types=0, index_nodes=0, dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)

            loss.backward()
            optimizer.step()

            if has_ghost:
                optimizer_ghost_particles.step()
                if (N > 0) & (N % 1000 == 0) & (train_config.ghost_method == 'MLP'):
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(to_numpy(ghosts_particles.data[run, :, 120, :].squeeze()))
                    fig.savefig(f"{log_dir}/tmp_training/embedding/ghosts_{N}.jpg", dpi=300)
                    plt.close()

            total_loss += loss.item()

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if has_cell_division:
            print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss_division / (N + 1) / n_particles / batch_size))
            logger.info("Epoch {}. Division Loss: {:.6f}".format(epoch, total_loss_division / (
                    N + 1) / n_particles / batch_size))
            torch.save({'model_state_dict': model_division.state_dict(),
                        'optimizer_state_dict': optimizer_division.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_division_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()},
                       os.path.join(log_dir, 'models', f'best_ghost_particles_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')

        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        if (simulation_config.n_interactions < 100) & (simulation_config.has_cell_division == False):
            ax = fig.add_subplot(1, 6, 3)
            if model_config.particle_model_name == 'PDE_G':
                rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
            elif model_config.particle_model_name == 'PDE_GS':
                rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
            else:
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            if dimension == 2:
                column_dimension = 5
            if dimension == 3:
                column_dimension = 7
            func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=True, config=config,
                                                                model_lin_edge=model.lin_edge, model_a=model.a,
                                                                n_nodes = 0,
                                                                dataset_number=1,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                types=to_numpy(x[:, column_dimension]),
                                                                cmap=cmap, dimension=dimension, device=device)
            ax = fig.add_subplot(1, 6, 4)
            match train_config.cluster_method:
                case 'kmeans_auto_plot':
                    labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
                case 'kmeans_auto_embedding':
                    labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                    proj_interaction = embedding
                case 'distance_plot':
                    labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
                case 'distance_embedding':
                    labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=1.5)
                    proj_interaction = embedding
                case 'distance_both':
                    new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                    labels, n_clusters = embedding_cluster.get(new_projection, 'distance')
            for n in range(n_clusters):
                pos = np.argwhere(labels == n)
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=5)
            label_list = []
            for n in range(n_particle_types):
                tmp = labels[index_particles[n]]
                label_list.append(np.round(np.median(tmp)))
            label_list = np.array(label_list)
            plt.xlabel('proj 0', fontsize=12)
            plt.ylabel('proj 1', fontsize=12)
            plt.text(0., 1.1, f'Nclusters: {n_clusters}', ha='left', va='top', transform=ax.transAxes)

            ax = fig.add_subplot(1, 6, 5)
            new_labels = labels.copy()
            for n in range(n_particle_types):
                new_labels[labels == label_list[n]] = n
                pos = np.argwhere(labels == label_list[n])
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                                color=cmap.color(n), s=0.1)
            Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            plt.text(0, 1.1, f'Accuracy: {np.round(Accuracy, 3)}', ha='left', va='top', transform=ax.transAxes,
                     fontsize=10)
            print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')
            logger.info(f'Accuracy: {np.round(Accuracy, 3)}    n_clusters: {n_clusters}')

            ax = fig.add_subplot(1, 6, 6)
            model_a_ = model.a[1].clone().detach()
            for n in range(n_clusters):
                pos = np.argwhere(labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    median_center = model_a_[pos, :]
                    median_center = torch.median(median_center, dim=0).values
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                    model_a_[pos, :] = median_center
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
            for n in np.unique(new_labels):
                pos = np.argwhere(new_labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (replace_with_cluster) & (
                    (epoch == 1 * n_epochs // 4) | (epoch == 2 * n_epochs // 4) | (epoch == 3 * n_epochs // 4)):
                match train_config.sparsity:
                    case 'replace_embedding':
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
                        if train_config.fix_cluster_embedding:
                            lr_embedding = 1E-8
                            lr = train_config.learning_rate_end
                            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                            logger.info(f'Learning rates: {lr}, {lr_embedding}')
                    case 'replace_embedding_function':
                        logger.info(f'replace_embedding_function')
                        # Constrain function domain
                        y_func_list = func_list * 0
                        for n in range(n_particles):
                            pos = np.argwhere(new_labels == n)
                            pos = pos.squeeze()
                            if pos.size > 0:
                                target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                                y_func_list[pos] = target_func
                        lr_embedding = 1E-8
                        if has_ghost:
                            lr = 1E-8
                        else:
                            lr = train_config.learning_rate_end
                        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                        for sub_epochs in range(20):
                            loss = 0
                            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                            pred = []
                            optimizer.zero_grad()
                            for n in range(n_particles):
                                embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                                    (1000, model_config.embedding_dim), device=device)
                                match model_config.particle_model_name:
                                    case 'PDE_A':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                    case 'PDE_A_bis':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_),
                                            dim=1)
                                    case 'PDE_B':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_G':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_E':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                                pred.append(model.lin_edge(in_features.float()))

                            pred = torch.stack(pred)
                            loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                            logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                            loss.backward()
                            optimizer.step()
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
            else:
                # if (epoch > n_epochs - 3) & (replace_with_cluster):
                #     lr_embedding = 1E-5
                #     lr = train_config.learning_rate_end
                #     optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                #     logger.info(f'Learning rates: {lr}, {lr_embedding}')
                if epoch > 3 * n_epochs // 4 + 1:
                    lr_embedding = train_config.learning_rate_embedding_end
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                    lr = train_config.learning_rate_start
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()



def data_train_particle_field(config, device):
    print('')

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    n_nodes = simulation_config.n_nodes
    n_node_types = simulation_config.n_node_types
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    has_large_range = train_config.large_range
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)

    l_dir, log_dir, logger = create_log_dir(config, dataset_name)
    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs}')
    logger.info(f'Graph files N: {NGraphs}')

    x_list = []
    y_list = []
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    x = x_list[0][0].clone().detach()
    y = y_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in trange(n_frames):
            if (k % 10 == 0) | (n_frames < 1000):
                x = torch.cat((x, x_list[run][k].clone().detach()), 0)
                y = torch.cat((y, y_list[run][k].clone().detach()), 0)
        print(x_list[run][k].shape)
        time.sleep(0.5)
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    vnorm = vnorm[4]
    ynorm = ynorm[4]
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')
    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(NGraphs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in range(n_frames):
            h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
    hnorm = torch.std(h)
    torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
    print(f'hnorm: {to_numpy(hnorm)}')
    logger.info(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    # mesh_pos = mesh_data['mesh_pos']
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']
    # face = mesh_data['face']
    mask_mesh = mask_mesh.repeat(batch_size, 1)
    h = []
    x = []
    y = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_0.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()
    if has_cell_division:
        model_division = Division_Predictor(config, device)
        optimizer_division, n_total_params_division = set_trainable_division_parameters(model_division, lr=1E-3)
        logger.info(f"Total Trainable Divsion Params: {n_total_params_division}")
        logger.info(f'Learning rates: 1E-3')

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()
    type_list = x[:, 5:6].clone().detach()
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, device)
        if train_config.ghost_method == 'MLP':
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.data], lr=5E-4)
        else:
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    index_nodes = []
    x_mesh = x_mesh_list[1][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == -n - 1)
        index_nodes.append(index.squeeze())

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    list_loss2 = []
    list_loss3 = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        if epoch == 1:
            repeat_factor = batch_size // old_batch_size
            mask_mesh = mask_mesh.repeat(repeat_factor, 1)
            if has_ghost:
                mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
                mask_ghost = np.tile(mask_ghost, batch_size)
                mask_ghost = np.argwhere(mask_ghost == 1)
                mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        total_loss_particle = 0
        total_loss_mesh = 0

        total_loss_division = 0

        Niter = n_frames * data_augmentation_loop // batch_size

        for N in range(Niter):

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            time_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)
                x = x_list[run][k].clone().detach()
                if has_ghost:
                    if train_config.ghost_method == 'MLP':
                        x_ghost = ghosts_particles.get_pos_t(dataset_id=run, frame=k)
                    else:
                        x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k)
                    x = torch.cat((x, x_ghost), 0)
                x_mesh = x_mesh_list[run][k].clone().detach()

                x_particle_field = torch.concatenate((x_mesh, x), 0)
                x_particle_field[:, 0] = torch.arange(x_mesh.shape[0] + x.shape[0], device=device)
                # compute connectivity rule
                distance = torch.sum(bc_dpos(
                    x_particle_field[:, None, 1:dimension + 1] - x_particle_field[None, :, 1:dimension + 1]) ** 2,
                                     dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_particle = adj_t.nonzero().t().contiguous()

                dataset = data.Data(x=x_particle_field, pos=x[:, 1:3], edge_index=edge_index,
                                    edge_particle=edge_particle, edge_mesh=mesh_data['edge_index'],
                                    edge_attr=mesh_data['edge_weight'])

                dataset_batch.append(dataset)
                y = y_list[run][k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)
                y = y / ynorm
                if data_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y

                y_mesh = y_mesh_list[run][k].clone().detach() / hnorm

                if batch == 0:
                    y_batch = y[:, 0:2]
                    y_mesh_batch = y_mesh[:, 0:1]

                else:
                    y_batch = torch.cat((y_batch, y), dim=0)
                    y_mesh_batch = torch.cat((y_mesh_batch, y_mesh), dim=0)


                if has_cell_division:
                    if batch == 0:
                        time_batch = torch.concatenate((x[:, 0:1], torch.ones_like(y[:, 3:4], device=device) * k),
                                                       dim=1)
                        y_batch_division = y[:, 2:3]
                    else:
                        time_batch = torch.concatenate((time_batch, torch.concatenate(
                            (x[:, 0:1], torch.ones_like(y[:, 2:3], device=device) * k), dim=1)), dim=0)
                        y_batch_division = torch.concatenate((y_batch_division, y[:, 3:4]), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()
            if has_ghost:
                optimizer_ghost_particles.zero_grad()
            if has_cell_division:
                optimizer_division.zero_grad()

            for batch in batch_loader:
                batch

            for batch in batch_loader:
                pred_y, pred_du = model(batch, data_id=run, training=True, vnorm=vnorm, phi=phi)

            if has_cell_division:
                pred_division = model_division(time_batch, data_id=run)
                loss_division = (pred_division - y_batch_division).norm(2)
                loss_division.backward()
                optimizer_division.step()
                total_loss_division += loss_division.item()

            if has_ghost:
                loss = ((pred_y[mask_ghost] - y_batch)).norm(2) + (pred_du - y_mesh_batch).norm(2)
            else:
                loss_particle = (pred_y - y_batch).norm(2)
                loss_mesh = (pred_du - y_mesh_batch).norm(2)
                loss = loss_particle + loss_mesh

            loss.backward()
            optimizer.step()

            if has_ghost:
                optimizer_ghost_particles.step()
                if (N > 0) & (N % 1000 == 0) & (train_config.ghost_method == 'MLP'):
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(to_numpy(ghosts_particles.data[run, :, 120, :].squeeze()))
                    fig.savefig(f"{log_dir}/tmp_training/embedding/ghosts_{N}.jpg", dpi=300)
                    plt.close()

            total_loss += loss.item()
            total_loss_particle += loss_particle.item()
            total_loss_mesh += loss_mesh.item()

            visualize_embedding = True

            if visualize_embedding & (((epoch == 0) & (N % 1000 == 0)) | (N == 0)):
                plot_training(config=config, dataset_name=dataset_name, model_name=model_config.particle_model_name,
                              log_dir=log_dir,
                              epoch=epoch, N=N, x=x, model=model, n_nodes=n_nodes, n_node_types=n_node_types, index_nodes=index_nodes, dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        print("Epoch {}. Loss_particle: {:.6f}".format(epoch, total_loss_particle / (N + 1) / n_particles / batch_size))
        print("Epoch {}. Loss_mesh: {:.6f}".format(epoch, total_loss_mesh / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss_particle: {:.6f}".format(epoch, total_loss_particle / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss_mesh: {:.6f}".format(epoch, total_loss_mesh / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        list_loss2.append(total_loss_particle / (N + 1) / n_particles / batch_size)
        list_loss3.append(total_loss_mesh / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        if has_cell_division:
            print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss_division / (N + 1) / n_particles / batch_size))
            logger.info("Epoch {}. Division Loss: {:.6f}".format(epoch, total_loss_division / (
                        N + 1) / n_particles / batch_size))
            torch.save({'model_state_dict': model_division.state_dict(),
                        'optimizer_state_dict': optimizer_division.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_division_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        if has_ghost:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()},
                       os.path.join(log_dir, 'models', f'best_ghost_particles_with_{NGraphs - 1}_graphs_{epoch}.pt'))

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')

        ax = fig.add_subplot(1, 6, 1)
        plt.plot(np.log(list_loss), color='k')
        plt.plot(np.log(list_loss2), color='r')
        plt.plot(np.log(list_loss3), color='g')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = get_embedding(model.a, 1)
        embedding = embedding[n_nodes:]
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        ax = fig.add_subplot(1, 6, 3)
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

        func_list = []
        popt_list = []
        for n in range(n_nodes):
            embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
            r = torch.tensor(np.linspace(-150, 150, 100)).to(device)
            in_features = torch.cat((r[:, None], embedding_), dim=1)
            h = model.lin_phi(in_features.float())
            popt, pcov = curve_fit(linear_model, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
            popt_list.append(popt)
            h = h[:, 0]
            func_list.append(h)
            if (n % 24):
                plt.plot(to_numpy(r),
                         to_numpy(h) * to_numpy(hnorm), linewidth=1,
                         color='k', alpha=0.05)
        func_list = torch.stack(func_list)
        coeff_norm = to_numpy(func_list)
        popt_list = np.array(popt_list)
        proj_interaction = popt_list
        proj_interaction[:, 1] = proj_interaction[:, 0]

        ax = fig.add_subplot(1, 6, 4)
        match train_config.cluster_method:
            case 'kmeans_auto_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
            case 'kmeans_auto_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                proj_interaction = embedding
            case 'distance_plot':
                labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
            case 'distance_embedding':
                labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=1.5)
                proj_interaction = embedding
            case 'distance_both':
                new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=5)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)

        plt.xlabel('proj 0', fontsize=12)
        plt.ylabel('proj 1', fontsize=12)
        plt.text(0., 1.1, f'Nclusters: {n_clusters}', ha='left', va='top', transform=ax.transAxes)

        ax = fig.add_subplot(1, 6, 5)
        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n
            pos = np.argwhere(labels == label_list[n])
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                            color=cmap.color(n), s=0.1)
        Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        plt.text(0, 1.1, f'Accuracy: {np.round(Accuracy, 3)}', ha='left', va='top', transform=ax.transAxes,
                 fontsize=10)
        print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')
        logger.info(f'Accuracy: {np.round(Accuracy, 3)}    n_clusters: {n_clusters}')

        ax = fig.add_subplot(1, 6, 6)
        model_a_ = model.a[1].clone().detach()
        for n in range(n_clusters):
            pos = np.argwhere(labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                median_center = model_a_[pos, :]
                median_center = torch.median(median_center, dim=0).values
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                model_a_[pos, :] = median_center
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
        for n in np.unique(new_labels):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
        plt.xlabel('ai0', fontsize=12)
        plt.ylabel('ai1', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)

        if (replace_with_cluster) & (
                (epoch == 1 * n_epochs // 4) | (epoch == 2 * n_epochs // 4) | (epoch == 3 * n_epochs // 4)):
            match train_config.sparsity:
                case 'replace_embedding':
                    # Constrain embedding domain
                    with torch.no_grad():
                        model.a[1] = model_a_.clone().detach()
                    print(f'regul_embedding: replaced')
                    logger.info(f'regul_embedding: replaced')
                    plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
                    if train_config.fix_cluster_embedding:
                        lr_embedding = 1E-8
                        lr = train_config.learning_rate_end
                        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                        logger.info(f'Learning rates: {lr}, {lr_embedding}')
                case 'replace_embedding_function':
                    logger.info(f'replace_embedding_function')
                    # Constrain function domain
                    y_func_list = func_list * 0
                    for n in range(n_nodes):
                        pos = np.argwhere(new_labels == n)
                        pos = pos.squeeze()
                        if pos.size > 0:
                            target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                            y_func_list[pos] = target_func
                    lr_embedding = 1E-8
                    if has_ghost:
                        lr = 1E-8
                    else:
                        lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    for sub_epochs in range(20):
                        loss = 0
                        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_nodes):
                            embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                                (1000, model_config.embedding_dim), device=device)
                            match model_config.particle_model_name:
                                case 'PDE_A':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                case 'PDE_A_bis':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, embedding_, embedding_),
                                        dim=1)
                                case 'PDE_B':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                case 'PDE_G':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                case 'PDE_E':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                            pred.append(model.lin_edge(in_features.float()))

                        pred = torch.stack(pred)
                        loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                        logger.info(f'    loss: {np.round(loss.item() / n_nodes, 3)}')
                        loss.backward()
                        optimizer.step()
                    # Constrain embedding domain
                    with torch.no_grad():
                        model.a[1] = model_a_.clone().detach()
                    print(f'regul_embedding: replaced')
                    logger.info(f'regul_embedding: replaced')
                    plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        else:
            if epoch > 3 * n_epochs // 4 + 1:
                lr_embedding = train_config.learning_rate_embedding_end
                lr = train_config.learning_rate_end
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                lr_embedding = train_config.learning_rate_embedding_start
                lr = train_config.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()



def data_train_mesh(config, device):

    print('')

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_nodes = simulation_config.n_nodes
    n_node_types = simulation_config.n_node_types
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    has_large_range = train_config.large_range
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)

    l_dir, log_dir, logger = create_log_dir(config, dataset_name)
    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs}')
    logger.info(f'Graph files N: {NGraphs}')

    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(NGraphs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()
    for run in range(NGraphs):
        for k in range(n_frames):
            h = torch.cat((h, y_mesh_list[run][k].clone().detach()), 0)
    hnorm = torch.std(h)
    torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
    print(f'hnorm: {to_numpy(hnorm)}')
    logger.info(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    h = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_7.pt"
    state_dict = torch.load(net,map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x_mesh = x_mesh_list[1][n_frames - 1].clone().detach()
    type_list = x_mesh[:, 5:6].clone().detach()
    n_nodes = x_mesh.shape[0]
    print(f'N nodes: {n_nodes}')
    logger.info(f'N nodes: {n_nodes}')

    index_nodes = []
    x_mesh = x_mesh_list[1][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == n)
        index_nodes.append(index.squeeze())

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if epoch == 1:
            repeat_factor = batch_size // old_batch_size
            mask_mesh = mask_mesh.repeat(repeat_factor, 1)

        total_loss = 0
        Niter = n_frames * data_augmentation_loop // batch_size
        if (batch_size == 1):
            Niter = Niter // 4

        for N in range(Niter):

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            for batch in range(batch_size):
                k = np.random.randint(n_frames - 1)
                x_mesh = x_mesh_list[run][k].clone().detach()
                if train_config.noise_level > 0:
                    x_mesh[:, 6:7] = x_mesh[:, 6:7] + train_config.noise_level * torch.randn_like(x_mesh[:, 6:7])
                dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
                dataset_batch.append(dataset)
                y = y_mesh_list[run][k].clone().detach() / hnorm
                if batch == 0:
                    y_batch = y
                else:
                    y_batch = torch.cat((y_batch, y), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=run)

            loss = ((pred - y_batch) * mask_mesh).norm(2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (((epoch == 0) & (N < 50000) & (N % 200 == 0)) | (N==0)):
                plot_training(config=config, dataset_name=dataset_name, model_name='WaveMesh',
                              log_dir=log_dir,
                              epoch=epoch, N=N, x=x_mesh, model=model, n_nodes=n_nodes, n_node_types=n_node_types, index_nodes=index_nodes, dataset_num=1,
                              index_particles=[], n_particles=[],
                              n_particle_types=[], ynorm=ynorm, cmap=cmap, axis=True, device=device)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_nodes / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_nodes / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_nodes / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')

        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_node_types):
            plt.scatter(embedding[index_nodes[n], 0],
                        embedding[index_nodes[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        if (simulation_config.n_interactions < 100):

            ax = fig.add_subplot(1, 6, 3)
            func_list = []
            popt_list = []
            for n in range(n_nodes):
                embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                if model_config.mesh_model_name == 'RD_RPS_Mesh':
                    embedding_ = model.a[1, n, :] * torch.ones((100, model_config.embedding_dim), device=device)
                    u = torch.tensor(np.linspace(0, 1, 100)).to(device)
                    u = u[:, None]
                    r = u
                    in_features = torch.cat((u, u, u, u, u, u, embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    h = h[:, 0]
                else:
                    r = torch.tensor(np.linspace(-150, 150, 100)).to(device)
                    in_features = torch.cat((r[:, None], embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    popt, pcov = curve_fit(linear_model, to_numpy(r.squeeze()), to_numpy(h.squeeze()))
                    popt_list.append(popt)
                    h = h[:, 0]
                func_list.append(h)
                if (n % 24):
                    plt.plot(to_numpy(r),
                             to_numpy(h) * to_numpy(hnorm), linewidth=1,
                             color='k', alpha=0.05)
            func_list = torch.stack(func_list)
            coeff_norm = to_numpy(func_list)
            popt_list = np.array(popt_list)
            if model_config.mesh_model_name == 'RD_RPS_Mesh':
                trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm)
                proj_interaction = trans.transform(coeff_norm)
            else:
                proj_interaction = popt_list
                proj_interaction[:, 1] = proj_interaction[:, 0]

            ax = fig.add_subplot(1, 6, 4)
            match train_config.cluster_method:
                case 'kmeans_auto_plot':
                    labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
                case 'kmeans_auto_embedding':
                    labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
                    proj_interaction = embedding
                case 'distance_plot':
                    labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')
                case 'distance_embedding':
                    labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=1.5)
                    proj_interaction = embedding
                case 'distance_both':
                    new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
                    labels, n_clusters = embedding_cluster.get(new_projection, 'distance')

            for n in range(n_clusters):
                pos = np.argwhere(labels == n)
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=5)
            label_list = []
            for n in range(n_node_types):
                tmp = labels[index_nodes[n]]
                label_list.append(np.round(np.median(tmp)))
            label_list = np.array(label_list)

            plt.xlabel('proj 0', fontsize=12)
            plt.ylabel('proj 1', fontsize=12)
            plt.text(0., 1.1, f'Nclusters: {n_clusters}', ha='left', va='top', transform=ax.transAxes)

            ax = fig.add_subplot(1, 6, 5)
            new_labels = labels.copy()
            for n in range(n_node_types):
                new_labels[labels == label_list[n]] = n
                pos = np.argwhere(labels == label_list[n])
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                                color=cmap.color(n), s=0.1)
            Accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            plt.text(0, 1.1, f'Accuracy: {np.round(Accuracy, 3)}', ha='left', va='top', transform=ax.transAxes,
                     fontsize=10)
            print(f'Accuracy: {np.round(Accuracy, 3)}   n_clusters: {n_clusters}')
            logger.info(f'Accuracy: {np.round(Accuracy, 3)}    n_clusters: {n_clusters}')

            ax = fig.add_subplot(1, 6, 6)
            model_a_ = model.a[1].clone().detach()
            for n in range(n_clusters):
                pos = np.argwhere(labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    median_center = model_a_[pos, :]
                    median_center = torch.median(median_center, dim=0).values
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                    model_a_[pos, :] = median_center
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='k')
            for n in np.unique(new_labels):
                pos = np.argwhere(new_labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), color='k', s=5)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (replace_with_cluster) & ((epoch == 1 * n_epochs // 4) | (epoch == 2 * n_epochs // 4) | (epoch == 3 * n_epochs // 4)):
                match train_config.sparsity:
                    case 'replace_embedding':
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
                        if train_config.fix_cluster_embedding:
                            lr_embedding = 1E-8
                            lr = train_config.learning_rate_end
                            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                            logger.info(f'Learning rates: {lr}, {lr_embedding}')
                    case 'replace_embedding_function':
                        logger.info(f'replace_embedding_function')
                        # Constrain function domain
                        y_func_list = func_list * 0
                        for n in range(n_nodes):
                            pos = np.argwhere(new_labels == n)
                            pos = pos.squeeze()
                            if pos.size > 0:
                                target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                                y_func_list[pos] = target_func
                        lr_embedding = 1E-8
                        if has_ghost:
                            lr = 1E-8
                        else:
                            lr = train_config.learning_rate_end
                        optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                        for sub_epochs in range(20):
                            loss = 0
                            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                            pred = []
                            optimizer.zero_grad()
                            for n in range(n_nodes):
                                embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                                    (1000, model_config.embedding_dim), device=device)
                                match model_config.particle_model_name:
                                    case 'PDE_A':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                    case 'PDE_A_bis':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_),
                                            dim=1)
                                    case 'PDE_B':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_G':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                    case 'PDE_E':
                                        in_features = torch.cat(
                                            (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                                pred.append(model.lin_edge(in_features.float()))

                            pred = torch.stack(pred)
                            loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                            logger.info(f'    loss: {np.round(loss.item() / n_nodes, 3)}')
                            loss.backward()
                            optimizer.step()
                        # Constrain embedding domain
                        with torch.no_grad():
                            model.a[1] = model_a_.clone().detach()
                        print(f'regul_embedding: replaced')
                        logger.info(f'regul_embedding: replaced')
                        plt.text(0, 1.1, f'Replaced', ha='left', va='top', transform=ax.transAxes, fontsize=10)
            else:
                if epoch > 3 * n_epochs // 4 + 1:
                    lr_embedding = train_config.learning_rate_embedding_end
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                    lr = train_config.learning_rate_start
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()


def data_train_signal(config, device):

    print('')

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    has_cell_division = simulation_config.has_cell_division
    data_augmentation = train_config.data_augmentation
    data_augmentation_loop = train_config.data_augmentation_loop
    target_batch_size = train_config.batch_size
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    replace_with_cluster = 'replace' in train_config.sparsity
    has_ghost = train_config.n_ghosts > 0
    has_large_range = train_config.large_range
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)

    l_dir, log_dir, logger = create_log_dir(config, dataset_name)
    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = len(graph_files)
    print(f'Graph files N: {NGraphs}')
    logger.info(f'Graph files N: {NGraphs}')

    x_list = []
    y_list = []
    for run in trange(NGraphs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{dataset_name}/models/best_model_with_1_graphs_4.pt"
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"Total Trainable Params: {n_total_params}")
    logger.info(f'Learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print('Update variables ...')
    # update variable if particle_dropout, cell_division, etc ...
    x = x_list[1][n_frames - 1].clone().detach()
    if dimension == 2:
        type_list = x[:, 5:6].clone().detach()
    else:
        type_list = x[:, 7:8].clone().detach()
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        else:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    mat = scipy.io.loadmat(simulation_config.connectivity_file)
    adjacency = torch.tensor(mat['A'], device=device)
    adj_t = adjacency > 0
    edge_index = adj_t.nonzero().t().contiguous()
    model.edges = edge_index

    print("Start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    time.sleep(1)
    for epoch in range(n_epochs + 1):

        old_batch_size = batch_size
        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')
        if epoch == 1:
            repeat_factor = batch_size // old_batch_size
            if has_mesh:
                mask_mesh = mask_mesh.repeat(repeat_factor, 1)
            if has_ghost:
                mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
                mask_ghost = np.tile(mask_ghost, batch_size)
                mask_ghost = np.argwhere(mask_ghost == 1)
                mask_ghost = mask_ghost[:, 0].astype(int)

        total_loss = 0
        total_loss_division = 0

        Niter = n_frames * data_augmentation_loop // batch_size
        if (has_mesh) & (batch_size == 1):
            Niter = Niter // 4

        for N in range(Niter):

            run = 1 + np.random.randint(NGraphs - 1)

            dataset_batch = []
            time_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)
                x = x_list[run][k].clone().detach()
                dataset = data.Data(x=x[:, :], edge_index=model.edges)
                dataset_batch.append(dataset)
                y = y_list[run][k].clone().detach()
                y = y / ynorm

                if batch == 0:
                    y_batch = y[:, 0:2]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:2]), dim=0)


            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=run)

            if not (has_large_range):
                loss = (pred - y_batch).norm(2)
            else:
                loss = ((pred - y_batch) / (y_batch)).norm(2) / 1E9

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            visualize_embedding = True
            if visualize_embedding & (epoch == 0) & (N < 10000) & (N % 200 == 0):
                plot_training(dataset_name=dataset_name, model_name=model_config.signal_model_name, log_dir=log_dir,
                              epoch=epoch, N=N, x=x, model=model, dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, device=device)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / (N + 1) / n_particles / batch_size))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{NGraphs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / (N + 1) / n_particles / batch_size)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # matplotlib.use("Qt5Agg")
        fig = plt.figure(figsize=(22, 4))
        # white background
        # plt.style.use('classic')

        ax = fig.add_subplot(1, 6, 1)
        plt.plot(list_loss, color='k')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        ax = fig.add_subplot(1, 6, 2)
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)

        ax = fig.add_subplot(1, 6, 3)
        gt_weight = to_numpy(adjacency[adj_t])
        pred_weight = to_numpy(model.weight_ij[adj_t])
        plt.scatter(gt_weight, pred_weight, s=0.1,c='k')
        plt.xlabel('gt weight', fontsize=12)
        plt.ylabel('predicted weight', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")
        plt.close()



def data_test(config, visualize=False, style='color', verbose=True, best_model=20, step=5, ratio=1, run=1, test_simulation=False, sample_embedding = False, device=[]):
    print('')

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_division = simulation_config.has_cell_division
    has_ghost = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension

    print(f'Test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)
    graph_files = glob.glob(f"graphs_data/graphs_{dataset_name}/x_list*")
    NGraphs = int(len(graph_files))
    if best_model == -1:
        net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs.pt"
    else:
        net = f"./log/try_{dataset_name}/models/best_model_with_{NGraphs - 1}_graphs_{best_model}.pt"
    print('Graph files N: ', NGraphs - 1)
    print(f'network: {net}')

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if verbose:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    if test_simulation==False:
        if has_mesh:
            mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            mesh_model.load_state_dict(state_dict['model_state_dict'])
            mesh_model.eval()
        else:
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            mesh_model = None
        if has_division:
            model_division = Division_Predictor(config, device)
            net = f"./log/try_{dataset_name}/models/best_model_division_with_{NGraphs - 1}_graphs_20.pt"
            state_dict = torch.load(net, map_location=device)
            model_division.load_state_dict(state_dict['model_state_dict'])
            model_division.eval()
        if os.path.isfile(os.path.join(log_dir, f'labels_{best_model}.pt')):
            print('Use learned labels')
            labels = torch.load(os.path.join(log_dir, f'labels_{best_model}.pt'))

    # nparticles larger than initially
    if ratio > 1:

        index_particles = []
        prev_index_particles = index

        new_nparticles = n_particles * ratio
        prev_nparticles = n_particles

        print('')
        print(f'New_number of particles: {new_nparticles}  ratio:{ratio}')
        print('')

        embedding = model.a[1].data.clone().detach()
        new_embedding = []
        new_labels = []

        for n in range(n_particle_types):
            for m in range(ratio):
                if (n == 0) & (m == 0):
                    new_embedding = embedding[prev_index_particles[n].astype(int), :]
                    new_labels = labels[prev_index_particles[n].astype(int)]
                else:
                    new_embedding = torch.cat((new_embedding, embedding[prev_index_particles[n].astype(int), :]), dim=0)
                    new_labels = torch.cat((new_labels, labels[prev_index_particles[n].astype(int)]), dim=0)

        model.a = nn.Parameter(
            torch.tensor(np.ones((NGraphs - 1, int(prev_nparticles) * ratio, 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        model.a.requires_grad = False
        model.a[1] = new_embedding
        n_particles = new_nparticles

        index_particles = []
        np_i = int(n_particles / n_particle_types)
        for n in range(n_particle_types):
            index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device))
    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)
    x = x_list[0][0].clone().detach()
    n_sub_population = n_particles // n_particle_types
    first_embedding = model.a[1].data.clone().detach()
    first_index_particles = []
    for n in range(n_particle_types):
        index = np.arange(n_particles * n // n_particle_types, n_particles * (n + 1) // n_particle_types)
        first_index_particles.append(index)
    n_particles = x.shape[0]
    config.simulation.n_particles = n_particles
    print(f'N particles: {n_particles}')
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    if sample_embedding:

        model_a_ = nn.Parameter(
            torch.tensor(np.ones((int(n_particles), model.embedding_dim)),
                         device=device,
                         requires_grad=False, dtype=torch.float32))
        for n in range(n_particles):
            t = to_numpy(x[n,5]).astype(int)
            index=first_index_particles[t][np.random.randint(n_sub_population)]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((model.n_dataset,int(n_particles), model.embedding_dim)),
                         device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_


    if has_ghost:
        model_ghost = Ghost_Particles(config, n_particles, vnorm, device)
        net = f"./log/try_{dataset_name}/models/best_ghost_particles_with_{NGraphs - 1}_graphs_20.pt"
        state_dict = torch.load(net, map_location=device)
        model_ghost.load_state_dict(state_dict['model_state_dict'])
        model_ghost.eval()
        x_removed_list = torch.load(f'graphs_data/graphs_{dataset_name}/x_removed_list_0.pt', map_location=device)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    if simulation_config.has_cell_division:
        cycle_length = torch.load(f'./graphs_data/graphs_{dataset_name}/cycle_length.pt', map_location=device).to(
            device)
        cycle_length_distrib = cycle_length[to_numpy(x[:, 5]).astype(int)].squeeze()
        A1 = torch.rand(cycle_length_distrib.shape[0], device=device)
        A1 = A1 * cycle_length_distrib
        A1 = A1[:, None]
    if has_mesh:
        hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_0.pt', map_location=device)
        mask_mesh = mesh_data['mask']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']

        xy = to_numpy(mesh_data['mesh_pos'])
        x_ = xy[:, 0]
        y_ = xy[:, 1]
        mask = to_numpy(mask_mesh)

        mask_mesh = (x_ > np.min(x_) + 0.02) & (x_ < np.max(x_) - 0.02) & (y_ > np.min(y_) + 0.02) & (
                    y_ < np.max(y_) - 0.02)
        mask_mesh = torch.tensor(mask_mesh, dtype=torch.bool, device=device)

        # plt.scatter(x_, y_, s=2, c=to_numpy(mask_mesh))
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    rmserr_list= []
    time.sleep(1)
    for it in trange(n_frames - 1):

        x0 = x_list[0][it].clone().detach()
        y0 = y_list[0][it].clone().detach()

        if model_config.signal_model_name == 'PDE_N':
            rmserr = torch.sqrt(torch.mean(torch.sum((x[:, 6:7] - x0[:, 6:7]) ** 2, axis=1)))
        else:
            rmserr = torch.sqrt(torch.mean(torch.sum((x[:, 1:3] - x0[:, 1:3]) ** 2, axis=1)))
        rmserr_list.append(rmserr.item())

        if has_mesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)

        if model_config.mesh_model_name == 'DiffMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config.mesh_model_name == 'WaveMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0)
            x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
            x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif model_config.mesh_model_name == 'RD_RPS_Mesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0)
                x[mask_mesh.squeeze(), 6:9] += pred[mask_mesh.squeeze()] * hnorm * delta_t
        else:

            x_ = x
            if has_ghost:
                x_ghost = model_ghost.get_pos(dataset_id=run, frame=it)
                x_ = torch.cat((x_, x_ghost), 0)

            if has_adjacency_matrix:
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            else:
                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

            if test_simulation:
                y = y0 / ynorm
            else:
                with torch.no_grad():
                    y = model(dataset, data_id=1, training=False, vnorm=vnorm,
                              phi=torch.zeros(1, device=device))  # acceleration estimation

            if has_ghost:
                y = y[mask_ghost]

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                if model_config.signal_model_name == 'PDE_N':
                    x[:, 6:7] += y * delta_t    # signal update
                else:
                    x[:, 3:5] = y

            if has_division:
                x_time = torch.concatenate((x[:, 0:1], torch.ones_like(x[:, 0:1], device=device) * it), dim=1)
                with torch.no_grad():
                    y_time = model_division(x_time, data_id=0)
                gt_y_time = y_list[0][it].clone().detach()

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)  # position update

        if simulation_config.has_cell_division:
            A1 = A1 + delta_t

        if (it % step == 0) & (it >= 0) & visualize:

            # plt.style.use('dark_background')

            # matplotlib.use("Qt5Agg")
            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})

            matplotlib.rcParams['savefig.pad_inches'] = 0

            fig = plt.figure(figsize=(12, 12))
            # if  (model_config.signal_model_name == 'PDE_N'):
            #     ax = fig.add_subplot(1, 1, 1)
            # elif (has_mesh | (simulation_config.boundary == 'periodic')) & (model_config.mesh_model_name != 'RD_RPS_Mesh'):
            #     ax = plt.axes([0, 0, 1, 1], frameon=False)
            # elif model_config.particle_model_name == 'PDE_G':
            #     ax = plt.axes([-2, -2, 2, 2], frameon=False)
            # elif model_config.particle_model_name == 'PDE_GS':
            #     ax = plt.axes([-0.5E10, -0.5E10, 0.5E10, 0.5E10], frameon=False)
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.autoscale(tight=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if has_mesh:
                pts = x[:, 1:3].detach().cpu().numpy()
                tri = Delaunay(pts)
                colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                if model_config.mesh_model_name == 'DiffMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                if model_config.mesh_model_name == 'WaveMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                if model_config.mesh_model_name == 'RD_Gray_Scott_Mesh':
                    fig = plt.figure(figsize=(12, 6))
                    ax = fig.add_subplot(1, 2, 1)
                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                    ax = fig.add_subplot(1, 2, 2)
                    colors = torch.sum(x[tri.simplices, 7], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                if model_config.mesh_model_name == 'RD_RPS_Mesh':
                    fig = plt.figure(figsize=(12, 12))
                    H1_IM = torch.reshape(x[:, 6:9], (100, 100, 3))
                    plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
            elif model_config.signal_model_name == 'PDE_N':
                # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=30, c=to_numpy(x[:,6]), cmap='viridis')
                plt.plot(to_numpy(x0[:, 6:7]), to_numpy(x[:, 6:7]), '.')
                plt.plot(to_numpy(y0[:, 0]), to_numpy(y[:, 0]), '.')
                plt.xlim([0,1])
                plt.ylim([0,1])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                # plt.autoscale(tight=True)
            else:
                s_p = 100
                if simulation_config.has_cell_division:
                    s_p = 25
                for n in range(n_particle_types):
                    plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                x[index_particles[n], 2].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
            if 'frame' in style:
                plt.xlabel(r'$x$', fontsize=64)
                plt.ylabel(r'$y$', fontsize=64)
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
            else:
                plt.xticks([])
                plt.yticks([])
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{dataset_name}_{it}.tif", dpi=170.7)
            plt.close()

            if has_ghost:
                fig = plt.figure(figsize=(12, 12))
                x_ghost_pos = bc_pos(x_ghost[:, 1:3])
                plt.scatter(x_ghost_pos[:, 0].detach().cpu().numpy(),
                            x_ghost_pos[:, 1].detach().cpu().numpy(), s=s_p, color='k')
                x_removed = x_removed_list[it]
                plt.scatter(x_removed[:, 1].detach().cpu().numpy(),
                            x_removed[:, 2].detach().cpu().numpy(), s=s_p / 2, color='g')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Ghost_{dataset_name}_{it}.tif", dpi=170.7)
                plt.close()

    print(f'RMSE = {np.round(np.mean(rmserr_list), 4)} +/- {np.round(np.std(rmserr_list), 4)}')

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0

    if n_particle_types>1000:
        n_particle_types = 3
    index_particles = []
    for n in range(n_particle_types):
        index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    x0_next = x_list[0][it + 1].clone().detach()
    temp1 = torch.cat((x, x0_next), 0)
    temp2 = torch.tensor(np.arange(n_particles), device=device)
    temp3 = torch.tensor(np.arange(n_particles) + n_particles, device=device)
    temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
    temp4 = torch.t(temp4)
    distance3 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
    p = torch.argwhere(distance3 < 0.3)
    pos = dict(enumerate(np.array((temp1[:, 1:3]).detach().cpu()), 0))
    dataset = data.Data(x=temp1[:, 1:3], edge_index=torch.squeeze(temp4[:, p]))
    vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
    nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False,ax=ax,edge_color='r', width=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    for n in range(n_particle_types):
        plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                    x[index_particles[n], 2].detach().cpu().numpy(), s=50, color=cmap.color(n))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel(r'$x$', fontsize=64)
    plt.ylabel(r'$y$', fontsize=64)
    # plt.text(0,0.9,f'RMS error: {np.round(np.mean(rmserr_list) * 100, 2)} +/- {np.round(np.std(rmserr_list) * 100, 2)} %')
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/rmserr_{dataset_name}_{it+2}.tif", dpi=170.7)



if __name__ == '__main__':

    config_list = ['arbitrary_3_dropout_10']


    for config_file in config_list:
        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        # print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        # data_generate(config, device=device, visualize=True, run_vizualized=0, style='graph', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 7)
        # data_generate_particle_field(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 20)
        data_train(config, device)
        # data_test(config, visualize=True, style='color', verbose=False, best_model=20, run=0, step=config.simulation.n_frames // 40, test_simulation=False, sample_embedding=True, device=device)


