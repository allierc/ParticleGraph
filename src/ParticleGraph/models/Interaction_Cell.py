import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
from ParticleGraph.models.Siren_Network import *


class Interaction_Cell(pyg.nn.MessagePassing):
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

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_Cell, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.n_particle_types = simulation_config.n_particle_types
        self.max_radius = simulation_config.max_radius
        self.data_augmentation = train_config.data_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.n_particles_max = simulation_config.n_particles_max
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.sigma = simulation_config.sigma
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.has_state = config.simulation.state_type != 'discrete'
        self.n_frames = simulation_config.n_frames
        self.do_tracking = train_config.do_tracking
        self.use_hot_encoding = train_config.use_hot_encoding


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device, initialisation='zeros')

        if self.do_tracking | self.has_state:
            self.a = nn.Parameter(torch.tensor(np.ones((self.n_particles_max, self.embedding_dim)), device=self.device, requires_grad=True, dtype=torch.float32))

            # self.cc = nn.Parameter(torch.tensor(np.zeros((self.embedding_dim)), device=self.device, requires_grad=False,
            #                                     dtype=torch.float32))
            # self.b = nn.Parameter(
            #     torch.tensor(np.zeros((self.n_particle_types, self.embedding_dim)), device=self.device,
            #                  requires_grad=False, dtype=torch.float32))
            # self.basis = nn.Parameter(torch.tensor(np.zeros((self.n_particle_types, self.embedding_dim)), device=self.device, requires_grad=False, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(np.ones((self.n_dataset, self.n_particles_max, 2)), device=self.device, requires_grad=True, dtype=torch.float32))





    def forward(self, data=[], data_id=[], training=[], vnorm=[], phi=[], has_field=False):

        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        self.has_field = has_field

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        area = x[:, 14:15]

        if self.do_tracking | self.has_state:
            particle_id = x[:, -1][:, None]
            if self.use_hot_encoding:
                # embedding = self.cc.clone().detach() + torch.matmul(self.a[to_numpy(particle_id), :], self.basis.clone().detach()).squeeze()
                embedding = torch.matmul(torch.sigmoid((self.a[to_numpy(particle_id), :]-0.5)*10), self.b.clone().detach()).squeeze()
            else:
                embedding = self.a[to_numpy(particle_id), :].squeeze()
        else:
            particle_id = x[:, 0:1]
            embedding = self.a[self.data_id, to_numpy(particle_id), :].squeeze()

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, particle_id=particle_id, embedding=embedding, field=field, area=area)

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, particle_id_i, particle_id_j, embedding_i, embedding_j, field_j, area_i, area_j):
        # distance normalized by the max radius
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm
        if self.dimension == 3:
            dpos_z_i = d_pos_i[:, 2] / self.vnorm
            dpos_z_j = d_pos_j[:, 2] / self.vnorm

        if self.data_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y
            new_dpos_x_i = self.cos_phi * dpos_x_i + self.sin_phi * dpos_y_i
            new_dpos_y_i = -self.sin_phi * dpos_x_i + self.cos_phi * dpos_y_i
            dpos_x_i = new_dpos_x_i
            dpos_y_i = new_dpos_y_i
            new_dpos_x_j = self.cos_phi * dpos_x_j + self.sin_phi * dpos_y_j
            new_dpos_y_j = -self.sin_phi * dpos_x_j + self.cos_phi * dpos_y_j
            dpos_x_j = new_dpos_x_j
            dpos_y_j = new_dpos_y_j

        match self.model:

            case 'PDE_Cell_A':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)

            case 'PDE_Cell_A_area':
                in_features = torch.cat((delta_pos, r[:, None], area_i * 1E3, area_j * 1E3, embedding_i, embedding_j), dim=-1)

            case 'PDE_Cell_B':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                                         dpos_y_j[:, None], embedding_i), dim=-1)

            case 'PDE_Cell_B_area':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                                         dpos_y_j[:, None], area_i, area_j, embedding_i, embedding_j), dim=-1)

        out = self.lin_edge(in_features) * field_j

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):

        if (self.model == 'PDE_A') | (self.model == 'PDE_Cell_A') | (self.model =='PDE_A_bis') | (self.model=='PDE_ParticleField_A'):
            return r * (p1[0] * torch.exp(-torch.abs(r) ** (2 * p1[1]) / (2 * self.sigma ** 2)) - p1[2] * torch.exp(-torch.abs(r) ** (2 * p1[3]) / (2 * self.sigma ** 2)))
        if (self.model == 'PDE_B') | (self.model == 'PDE_Cell_B'):
            cohesion = p1[0] * 0.5E-5 * r
            separation = -p1[2] * 1E-8 / r
            return (cohesion + separation) * p1[1] / 500
        if self.model == 'PDE_G':
            psi = p1 / r ** 2
            return psi[:, None]
        if self.model == 'PDE_E':
            acc = p1 * p2 / r ** 2
            return -acc  # Elec particles
