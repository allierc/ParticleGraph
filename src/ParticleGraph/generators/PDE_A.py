import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], sigma=[], bc_dpos=[]):
        super(PDE_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.sigma = sigma
        self.bc_dpos = bc_dpos

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        parameters = self.p[particle_type,:]
        d_pos = self.propagate(edge_index, pos=x[:, 1:3], parameters=parameters, sigma=self.sigma)
        return d_pos

    def message(self, pos_i, pos_j, parameters_i):
        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # squared distance
        psi = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
               - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
        d_pos = psi[:, None] * self.bc_dpos(pos_j - pos_i)
        return d_pos

    def psi(self, r, p):
        return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2))
                    - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))
