import torch
import torch.nn as nn
import torch_geometric as pyg
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
import numpy as np

class Signal_Propagation4(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the first derivative of a scalar field on a mesh.
    The node embedding is defined by a table self.a
    Note the Laplacian coeeficients are in data.edge_attr

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the first derivative of a scalar field on a mesh (dimension 3).
    """

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Signal_Propagation4, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.embedding_dim = model_config.embedding_dim
        self.n_particles = simulation_config.n_particles
        self.n_dataset = config.training.n_runs
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update
        self.bc_dpos = bc_dpos
        self.adjacency_matrix = simulation_config.adjacency_matrix

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size, nlayers=self.n_layers_update,
                            hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(torch.ones((self.n_dataset,int(self.n_particles), self.embedding_dim), device=self.device, requires_grad=True, dtype=torch.float32))

        self.W = nn.Parameter(torch.randn((int(self.n_particles),int(self.n_particles)), device=self.device, requires_grad=True, dtype=torch.float32))

        self.mask = torch.ones((int(self.n_particles),int(self.n_particles)), device=self.device, requires_grad=False, dtype=torch.float32)
        self.mask.fill_diagonal_(0)


    def forward(self, data=[], data_id=[], return_all=False, excitation=[]):
        self.data_id = data_id
        x, edge_index = data.x, data.edge_index

        u = data.x[:, 6:7]
        particle_id = to_numpy(x[:, 0])
        embedding = self.a[1, particle_id, :]

        # msg = torch.matmul(self.W * self.mask, self.lin_edge(u))
        # col = torch.cat((u, torch.ones_like(u), torch.ones_like(u), embedding), dim=1)
        # T = self.W * self.mask
        # msg = torch.zeros_like(u)
        # for i in range(self.n_particles):
        #     col[:,1:3] = self.a[1, i, :] * torch.ones((self.n_particles, 2), device=self.device)
        #     msg[i] = torch.sum(T[i] * self.lin_edge(col).squeeze())

        msg = self.propagate(edge_index, u=u, embedding=embedding)
        in_features = torch.cat([u, embedding], dim=1)
        pred = self.lin_phi(in_features) + msg

        return pred

    def message(self, edge_index_i, edge_index_j, u_j, embedding_i, embedding_j):

        T = self.W * self.mask
        in_features = torch.cat([u_j, embedding_i, embedding_j], dim=1)
        index_i = to_numpy(edge_index_i)
        index_j = to_numpy(edge_index_i)

        return T[index_i,index_j][:,None] * self.lin_edge(in_features)

    def update(self, aggr_out):
        return aggr_out

    def psi(self, r, p):
        return p * r