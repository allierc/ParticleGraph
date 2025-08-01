
import umap
import torch
from ParticleGraph.models.MLP import MLP
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *

# from ParticleGraph.models.utils import reparameterize
# from ParticleGraph.models.Siren_Network import Siren
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



def density_laplace(y, x):
    grad = density_gradient(y, x)
    return density_divergence(grad, x)
def density_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div
def density_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad






class Operator_smooth(pyg.nn.MessagePassing):

    """
    Model learning kernel operators.
    The methods follows the particle smoothing techniques proposed in the paper:
    'Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids'

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the kernel operators and their convolution with the data
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2, model_density=[]):

        super(Operator_smooth, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers
        self.n_particles = simulation_config.n_particles
        self.n_particles_max = simulation_config.n_particles_max
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.noise_model_level = train_config.noise_model_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.model_type = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.time_window = train_config.time_window
        self.model_density = model_density
        self.sub_sampling = simulation_config.sub_sampling
        self.prediction = model_config.prediction
        self.kernel_var = self.max_radius ** 2

        self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))
        self.field_type = model_config.field_type

        if self.update_type == 'pre_mlp':
            self.pre_lin_edge = MLP(input_size=self.pre_input_size, output_size=self.pre_output_size, nlayers=self.pre_n_layers,
                                hidden_size=self.pre_hidden_dim, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if 'mlp' in self.update_type:
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles_max) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

        self.siren = Siren_Network(image_width=100, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr,
                                hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                hidden_omega_0=model_config.omega )

    def forward(self, data=[], data_id=[], training=[], phi=[], continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_id = x[:, 0:1].long()
        embedding = self.a[data_id, particle_id, :].squeeze()
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        density_null = torch.zeros((pos.shape[0], 2), device=self.device)
        if continuous_field:
            self.mode = 'pre_mlp'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            density = torch.zeros((pos.shape[0], 1), device=self.device)
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density)
        else:
            self.mode = 'pre_mlp'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=density_null)
            self.mode = 'mlp'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, embedding=embedding, density=self.density)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, embedding_i, embedding_j, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'pre_mlp':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True

            density_kernel = torch.exp(-(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:,None]

            self.modulation = self.siren(coords=mgrid) * max_radius **2
            kernel_modified = torch.exp(-2*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None] * self.modulation

            grad_autograd = -density_gradient(kernel_modified, mgrid)
            laplace_autograd = density_laplace(kernel_modified, mgrid)

            self.kernel_operators = torch.cat((kernel_modified, grad_autograd, laplace_autograd), dim=-1)

            return density_kernel

            # kernel_modified = torch.exp(-2 * (mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / (20*self.kernel_var))[:, None]
            # fig = plt.figure(figsize=(6, 6))
            # plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=10, c=to_numpy(kernel_modified))
            # plt.show()

        else:
            # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
            # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
            # out = field_j * self.kernel_operators[:, 1:2] / density_j

            grad_density = self.kernel_operators[:, 1:3]  # d_rho_x d_rho_y

            # velocity = self.kernel_operators[:, 0:1] * torch.sum(d_pos_j**2, dim=1)[:,None] / density_j
            # grad_velocity = self.kernel_operators[:, 1:3] * torch.sum(d_pos_j**2, dim=1)[:,None].repeat(1,2) / density_j.repeat(1,2)
            # out = torch.cat((grad_density, velocity, grad_velocity), dim = 1) # d_rho_x d_rho_y, velocity
            # out = field_j * self.kernel_operators[:, 1:2] / density_j  # grad_x

            if 'laplacian' in self.field_type:
                out = field_j * self.kernel_operators[:, 3:4] / density_j  # laplacian
            elif 'grad_density' in self.field_type:
                out = grad_density
            else:
                out = grad_density

            return out


        fig = plt.figure(figsize=(6, 6))
        plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=100, c=to_numpy(self.kernel_operators[:,3:4]))

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(to_numpy(mgrid[:,0]), to_numpy(mgrid[:,1]), s=100, c=to_numpy(self.pre_lin_edge(mgrid)))


    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)



def arbitrary_gaussian_grad_laplace (mgrid, n_gaussian, device):

    mgrid.requires_grad = True
    x = mgrid[:, 0]
    y = mgrid[:, 1]
    size = np.sqrt(mgrid.shape[0]).astype(int)

    u = torch.zeros(mgrid.shape[0], device=device)

    for k in range(n_gaussian):
        x0 = np.random.uniform(0, 1)
        y0 = np.random.uniform(0, 1)
        a = np.random.uniform(0, 1)
        sigma = 1/2 * np.random.uniform(0.05, 0.1)
        u = u + a * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    grad_autograd = density_gradient(u, mgrid)
    laplacian_autograd = density_laplace(u, mgrid)

    return u, grad_autograd, laplacian_autograd

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131)
    plt.imshow(to_numpy(u).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('u(x,y)')
    ax = fig.add_subplot(132)
    plt.imshow(to_numpy(grad_autograd[:,0]).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('Grad_x(u(x,y)) autograd')
    ax = fig.add_subplot(133)
    plt.imshow(to_numpy(laplacian_autograd).reshape(size,size), cmap='viridis', extent=[0, 1, 0, 1])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title('Laplacian(u(x,y)) autograd')
    plt.show()

def remove_files_from_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it is a file and remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it is a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')




if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import torch_geometric.data as data
    from ParticleGraph.utils import choose_boundary_values
    from ParticleGraph.config import ParticleGraphConfig
    import os
    import shutil
    from torch_geometric.loader import DataLoader

    remove_files_from_folder('tmp')

    mode = 'gaussian'

    if mode == 'gaussian':
        config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/test_smooth_particle.yaml')
    elif mode == 'wave':
        # config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/wave/wave_smooth_particle.yaml')
        # config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/wave/wave_smooth_particle_1.yaml')
        config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/wave/wave_smooth_particle_2.yaml')
    elif mode == 'cell_MDCK':
        config = ParticleGraphConfig.from_yaml('/groups/saalfeld/home/allierc/Py/ParticleGraph/config/cell/cell_MDCK_4.yaml')

    device = 'cuda:0'
    dimension = 2
    bc_pos, bc_dpos = choose_boundary_values('no')
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    lr = config.training.learning_rate_start
    batch_size = config.training.batch_size

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    plt.style.use('dark_background')

    model = Operator_smooth(config=config, device=device, aggr_type='add', bc_dpos=bc_dpos, dimension=dimension)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    phi = torch.zeros(1, device=device)
    threshold = 0.05


    if mode == 'gaussian':
        tensors = tuple(dimension * [torch.linspace(0, 1, steps=100)])
        x = torch.stack(torch.meshgrid(*tensors), dim=-1)
        x = x.reshape(-1, dimension)
        x = torch.cat((torch.arange(x.shape[0])[:, None], x, torch.zeros((x.shape[0], 9))), 1)
        x = x.to(device)
        x.requires_grad = False
        size = np.sqrt(x.shape[0]).astype(int)
        x0 = x
    elif mode=='wave':
        dataset_name = config.dataset
        n_frames = config.simulation.n_frames
        x_mesh_list = []
        y_mesh_list = []
        x_mesh = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_{dataset_name}/x_mesh_list_0.pt', map_location=device, weights_only=True)
        x_mesh_list.append(x_mesh)
        y_mesh = torch.load(f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_{dataset_name}/y_mesh_list_0.pt', map_location=device, weights_only=True)
        y_mesh_list.append(y_mesh)

    for epoch in range(0, 5000):
        if mode == 'gaussian':

            dataset_batch = []

            for batch in range(batch_size):

                x = x0.clone().detach() + 0.05 * torch.randn_like(x0)
                # x = x[torch.randperm(x.size(0))[:int(0.5 * x.size(0))]] # removal of 10%
                u, grad_u, laplace_u = arbitrary_gaussian_grad_laplace(mgrid=x[:, 1:3], n_gaussian=5, device=device)
                L_u = laplace_u.clone().detach()
                x[:, 6:7] = u[:, None].clone().detach()

                # discrete_pos = torch.argwhere((u >= threshold) | (u <= -threshold))
                # x = x[discrete_pos].squeeze()
                # L_u = L_u[discrete_pos].squeeze()

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)
                dataset_batch.append(dataset)

                if batch == 0:
                    data_id = torch.ones((x.shape[0], 1), dtype=torch.int)
                    y_batch = L_u
                else:
                    data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int)), dim=0)
                    y_batch = torch.cat((y_batch, L_u), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=data_id, training=True, phi=phi)

            loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()

            # u = u[discrete_pos]
            # grad_u = grad_u[discrete_pos]
            # laplace_u = laplace_u[discrete_pos]

            print(epoch, loss)

            if epoch % 10 == 0:
                pred = pred[-x.shape[0]:]

                fig = plt.figure(figsize=(16, 6))
                ax = fig.add_subplot(241)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=1, c='w')
                pos = torch.argwhere(edge_index[0, :] == 350)
                plt.scatter(to_numpy(x[edge_index[1, pos], 1]), to_numpy(x[edge_index[1, pos], 2]), s=10, c='r')
                ax.invert_yaxis()
                plt.title('density')
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                ax = fig.add_subplot(242)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=2, c=to_numpy(u))
                ax.invert_yaxis()
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.title('u')
                ax = fig.add_subplot(243)
                # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=2, c=to_numpy(L_u[:,0]))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=2, c=to_numpy(L_u), vmin=-200, vmax=10)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.colorbar()
                ax.invert_yaxis()
                plt.title('true L_u')
                ax = fig.add_subplot(244)
                # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=2, c=to_numpy(pred))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=2, c=to_numpy(pred), vmin=-200, vmax=10)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.colorbar()
                ax.invert_yaxis()
                plt.title('pred L_u')
                ax = fig.add_subplot(245)
                indices = torch.randperm(x.shape[0])[:1000]
                plt.scatter(to_numpy(model.delta_pos[indices, 0]), to_numpy(model.delta_pos[indices, 1]), s=50,
                            c=to_numpy(model.modulation[indices]))
                plt.xlim([-max_radius, max_radius])
                plt.ylim([-max_radius, max_radius])
                plt.colorbar()
                ax = fig.add_subplot(246)
                plt.scatter(to_numpy(model.delta_pos[indices, 0]), to_numpy(model.delta_pos[indices, 1]), s=50,
                            c=to_numpy(model.kernel_operators[indices, 3:4]))
                plt.xlim([-max_radius, max_radius])
                plt.ylim([-max_radius, max_radius])
                plt.colorbar()
                plt.tight_layout()
                # plt.show()
                plt.savefig(f'tmp/learning_{epoch}.tif')
                plt.close()

            # # matplotlib.use("Qt5Agg")
            # fig = plt.figure(figsize=(12, 3))
            # ax = fig.add_subplot(141)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
            #             c=to_numpy(model.kernel_operators[:, 0:1]))
            # plt.title('kernel')
            # ax = fig.add_subplot(142)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
            #             c=to_numpy(model.kernel_operators[:, 1:2]))
            # plt.title('grad_x')
            # ax = fig.add_subplot(143)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
            #             c=to_numpy(model.kernel_operators[:, 2:3]))
            # plt.title('grad_y')
            # ax = fig.add_subplot(144)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
            #             c=to_numpy(model.kernel_operators[:, 3:4]))
            # plt.title('laplace')
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(f'tmp/kernels_{epoch}.tif')
            # plt.close()
        elif mode == 'wave':

            dataset_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)
                x = x_mesh_list[0][k].clone().detach()
                L_u = y_mesh_list[0][k].clone().detach()
                L_u = torch.where(torch.isnan(L_u), torch.zeros_like(L_u), L_u)
                u = x[:, 6:7].clone().detach()

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance >= 0)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, pos=x[:, 1:dimension + 1], edge_index=edge_index)

                dataset_batch.append(dataset)

                if batch == 0:
                    data_id = torch.ones((x.shape[0], 1), dtype=torch.int)
                    y_batch = L_u
                else:
                    data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int)), dim=0)
                    y_batch = torch.cat((y_batch, L_u), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=data_id, training=True, phi=phi)

            loss = (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()

            print(epoch, loss)

            if epoch % 10 == 0:
                pred = pred[-x.shape[0]:]

                fig = plt.figure(figsize=(24, 12.4))
                ax = fig.add_subplot(241)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=0.5, c='w')
                pos = torch.argwhere(edge_index[0, :] == 5050)
                plt.scatter(to_numpy(x[edge_index[1, pos], 1]), to_numpy(x[edge_index[1, pos], 2]), s=2, c='r')
                ax.invert_yaxis()
                # plt.colorbar()
                plt.title('nodes', fontsize=16)
                ax = fig.add_subplot(245)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=50, c='w')
                pos = torch.argwhere(edge_index[0, :] == 5050)
                plt.scatter(to_numpy(x[edge_index[1, pos], 1]), to_numpy(x[edge_index[1, pos], 2]), s=50, c='r')
                cx = to_numpy(x[edge_index[0, pos[0]], 1])
                cy = to_numpy(x[edge_index[0, pos[0]], 2])
                plt.xlim([cx - 2 * max_radius, cx + 2 * max_radius])
                plt.ylim([cy - 2 * max_radius, cy + 2 * max_radius])
                ax.invert_yaxis()
                # plt.colorbar()
                plt.title('nodes', fontsize=16)
                ax = fig.add_subplot(242)
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=2, c=to_numpy(u), vmin=-1000, vmax=1000)
                ax.invert_yaxis()
                # plt.colorbar()
                plt.title('signal u', fontsize=16)
                ax = fig.add_subplot(243)
                # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=2, c=to_numpy(L_u[:,0]))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=2, c=to_numpy(L_u), vmin=-30, vmax=30)
                # plt.colorbar()
                ax.invert_yaxis()
                plt.title('true voronoi Laplacian(u) - Voronoi', fontsize=16)
                ax = fig.add_subplot(244)
                # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=2, c=to_numpy(pred))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=2, c=to_numpy(pred), vmin=-30, vmax=30)
                # plt.colorbar()
                ax.invert_yaxis()
                plt.title('learned Laplacian(u)', fontsize=16)

                ax = fig.add_subplot(246)
                indices = torch.randperm(x.shape[0])[:1000]
                plt.scatter(to_numpy(model.delta_pos[indices, 0]), to_numpy(model.delta_pos[indices, 1]), s=10,
                            c=to_numpy(model.modulation[indices]))
                plt.xlim([-max_radius, max_radius])
                plt.ylim([-max_radius, max_radius])
                # plt.colorbar()
                plt.title('NNR modulation', fontsize=16)
                ax = fig.add_subplot(247)
                plt.scatter(to_numpy(model.delta_pos[indices, 0]), to_numpy(model.delta_pos[indices, 1]), s=10,
                            c=to_numpy(model.kernel_operators[indices, 3:4]))
                plt.xlim([-max_radius, max_radius])
                plt.ylim([-max_radius, max_radius])
                # plt.colorbar()
                plt.title('learned kernel operator', fontsize=16)
                ax = fig.add_subplot(248)
                # plt.scatter(to_numpy(x[:,1]), to_numpy(x[:,2]), s=2, c=to_numpy(pred))
                plt.scatter(to_numpy(L_u), to_numpy(pred), s=1, c='w', alpha=0.5)
                plt.xlabel('true Laplacian(u)', fontsize=16)
                plt.ylabel('learned Laplacian(u)', fontsize=16)
                plt.xlim([-30, 30])
                plt.ylim([-30, 30])
                plt.tight_layout()
                plt.savefig(f'tmp/learning_{epoch}.tif')
                plt.close()

            # # matplotlib.use("Qt5Agg")
            # fig = plt.figure(figsize=(22, 5))
            # ax = fig.add_subplot(141)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=100,
            #             c=to_numpy(model.kernel_operators[:, 0:1]))
            # plt.title('kernel')
            # ax = fig.add_subplot(142)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=100,
            #             c=to_numpy(model.kernel_operators[:, 1:2]))
            # plt.title('grad_x')
            # ax = fig.add_subplot(143)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=100,
            #             c=to_numpy(model.kernel_operators[:, 2:3]))
            # plt.title('grad_y')
            # ax = fig.add_subplot(144)
            # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=100,
            #             c=to_numpy(model.kernel_operators[:, 3:4]))
            # plt.title('laplace')
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(f'tmp/kernels_{epoch}.tif')
            # plt.close()



























