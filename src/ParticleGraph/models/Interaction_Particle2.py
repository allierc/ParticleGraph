import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
# from ParticleGraph.models.utils import reparameterize


class Interaction_Particle2(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2, model_density=[]):

        super(Interaction_Particle2, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.model = model_config.particle_model_name

        self.pre_input_size = model_config.pre_input_size
        self.pre_output_size = model_config.pre_output_size
        self.pre_hidden_dim = model_config.pre_hidden_dim
        self.pre_n_layers = model_config.pre_n_layers

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers
        
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update

        self.n_dataset = train_config.n_runs
        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim
        self.dimension = dimension
        self.delta_t = simulation_config.delta_t
        self.max_radius = simulation_config.max_radius
        self.bc_dpos = bc_dpos
        self.remove_self = train_config.remove_self


        self.time_window = train_config.time_window
        self.noise_model_level = train_config.noise_model_level
        self.sub_sampling = simulation_config.sub_sampling
        self.prediction = model_config.prediction


        if self.model == 'PDE_M2':
            self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

            self.lin_edge2 = MLP(input_size=self.output_size + 4, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)
        else:
            self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update, nlayers=self.n_layers_update,
                                    hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles), self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

    def forward(self, data=[], data_id=[], training=[], phi=[], has_field=False, k = 0):

        x, edge_index = data.x, data.edge_index
        if self.remove_self:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.time_window == 0:
            particle_id = x[:, 0:1].long()
            embedding = self.a[data_id, particle_id, :].squeeze()
            pos = x[:, 1:self.dimension+1]
        else:
            particle_id = x[0][:, 0:1].long()
            embedding = self.a[data_id, particle_id, :].squeeze()
            x = torch.stack(x)
            pos = x[:, :, 1:self.dimension + 1]
            pos = pos.transpose(0, 1)
            pos = torch.reshape(pos, (pos.shape[0], pos.shape[1] * pos.shape[2]))

        if training & (self.noise_model_level > 0):
            noise = torch.randn_like(pos) * self.noise_model_level
            pos = pos + noise

        if self.model == 'PDE_M2':
            self.step = 0
            pred = self.propagate(edge_index=edge_index, pos=pos, embedding=embedding)
            self.step = 1
            pred = self.propagate(edge_index=edge_index, pos=pred, embedding=embedding)

        else:
            self.step = 0
            pred = self.propagate(edge_index=edge_index, pos=pos, embedding=embedding)

        if self.update_type == 'mlp':
            pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
            out = self.lin_phi(torch.cat((pred, embedding, pos_p), dim=-1))
        else:
            out = pred

        if self.sub_sampling>1:
            pred = out
            d_pos = x[:, :, self.dimension + 1:1 + 2 * self.dimension]
            d_pos = d_pos.transpose(0, 1)
            d_pos = torch.reshape(d_pos, (d_pos.shape[0], d_pos.shape[1] * d_pos.shape[2]))
            for k in range(self.sub_sampling):
                if self.prediction == '2nd_derivative':
                    y = pred * self.ynorm * self.delta_t / self.sub_sampling
                    d_pos = d_pos + y  # speed update
                else:
                    y = pred * self.vnorm
                    d_pos = y
                pos = pos + d_pos * self.delta_t / self.sub_sampling
                out = pos
                if self.update_type == 'mlp':
                    pos_p = (pos - pos[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                    pred = self.lin_phi(torch.cat((self.propagate(edge_index, pos=pos, embedding=embedding), embedding, pos_p), dim=-1))
                else:
                    pred = self.propagate(edge_index, pos=pos, embedding=embedding)

        return out


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, embedding_i, embedding_j):

        if self.step == 0:
            if self.time_window == 0:
                delta_pos = self.bc_dpos(pos_j - pos_i)
                in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)
            else:
                pos_i_p = (pos_i - pos_i[:, 0:self.dimension].repeat(1, self.time_window))[:, self.dimension:]
                pos_j_p = (pos_j - pos_i[:, 0:self.dimension].repeat(1, self.time_window))
                in_features = torch.cat((pos_i_p, pos_j_p, embedding_i, embedding_j), dim=-1)
            out = self.lin_edge(in_features)
            return out

        elif self.step == 1:
            delta_pos = self.bc_dpos(pos_j - pos_i)
            in_features = torch.cat((delta_pos, embedding_i, embedding_j), dim=-1)
            out = self.lin_edge2(in_features)
            return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


