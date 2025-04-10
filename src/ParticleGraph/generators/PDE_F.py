import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import *


class PDE_F(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of fluidic particles.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=None, bc_dpos=None, dimension=2, delta_t=0.1, max_radius=0.05, field_type=None):
        super(PDE_F, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.field_type = field_type
        self.p = p[0]
        self.dimension = dimension
        self.delta_t = delta_t
        self.max_radius = max_radius
        self.bc_dpos = bc_dpos

        self.kernel_var = self.max_radius ** 2
        # self.kernel_norm = np.pi * self.kernel_var * (1 - np.exp(-self.max_radius ** 2/ self.kernel_var))
        # self.kernel_norm = 2


    def forward(self, data, continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = to_numpy(x[:, 1 + 2*self.dimension])
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]

        pos = pos + 1 * d_pos * self.delta_t

        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        if continuous_field:
            self.mode = 'kernel'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=torch.zeros_like(x[:, 0:1]))
            density = torch.zeros_like(x[:, 0:1])
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=density)
        else:
            self.mode = 'kernel'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=torch.zeros_like(x[:, 0:1]))
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, particle_type=particle_type, density=self.density)

        # out[:,0:self.dimension] = out[:,0:self.dimension] / (self.density.repeat(1,self.dimension) + 1E-7)        # divide force by density

        # out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
        # out = torch.where(torch.isnan(out), torch.zeros_like(out), out)

        out[:, 1] = out[:, 1] + torch.ones_like(out[:, 1]) * self.p[0] * 9.8

        # if torch.isnan(out).any():
        #     print('nan')
        # if torch.isinf(out).any():
        #     print('inf')


        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, particle_type_i, particle_type, density_i, density_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'kernel':
            self.mgrid = delta_pos.clone().detach()
            self.mgrid.requires_grad = True
            density_kernel = torch.exp(-4*(self.mgrid[:, 0] ** 2 + self.mgrid[:, 1] ** 2) / self.kernel_var)[:, None] / 2

            if 'gaussian' in self.field_type:
                pressure_kernel = density_kernel
            elif 'triangle' in self.field_type:
                dist = torch.sqrt(torch.sum(self.mgrid ** 2, dim=1))
                pressure_kernel = ((self.max_radius - dist)**2 / self.kernel_var)[:, None] / 1.309

            grad_density_kernel = density_gradient(density_kernel, self.mgrid)
            grad_pressure_kernel = density_gradient(pressure_kernel, self.mgrid)
            grad_pressure_kernel = torch.where(torch.isnan(grad_pressure_kernel), torch.zeros_like(grad_pressure_kernel), grad_pressure_kernel)
            laplace_autograd = density_laplace(density_kernel, self.mgrid)

            self.kernel_operators = torch.cat((density_kernel, grad_density_kernel, grad_pressure_kernel, laplace_autograd), dim=-1)

            return density_kernel

        elif self.mode == 'message_passing':

            pressure_force = torch.relu((density_i+density_j)/2 - self.p[2]/2) * self.kernel_operators[:, 3:5] * self.p[1] / density_j.repeat(1,2)
            viscosity_force = (d_pos_j-d_pos_i) * self.p[3] * self.kernel_operators[:, 0:1].repeat(1,2) / density_j.repeat(1,2)

            # viscosity_force = d_pos_j * self.p[3] * self.kernel_operators[:, 5:7] / density_j.repeat(1,2)
            # convection_force_x = d_pos_i[:,0:1] * self.kernel_operators[:, 1:2] *  d_pos_j[:,0:1] + d_pos_i[:,1:2] * self.kernel_operators[:, 2:3] *  d_pos_j[:,0:1]
            # convection_force_y = d_pos_i[:, 0:1] * self.kernel_operators[:, 1:2] * d_pos_j[:, 1:2] + d_pos_i[:,1:2] * self.kernel_operators[:, 2:3] * d_pos_j[:, 1:2]
            # convection_force = -torch.cat((convection_force_x, convection_force_y), dim=1) / density_j.repeat(1,2) * self.p[4]

            force = pressure_force + viscosity_force # + convection_force

            velocity = self.kernel_operators[:, 0:1] * torch.sum(d_pos_j ** 2, dim=1)[:, None] / density_j

            out = torch.cat((force / density_i, velocity), dim=1)


            return out











        # out = self.lin_edge(field_j) * self.kernel_operators[:,1:2] / density_j
        # out = self.lin_edge(field_j) * self.kernel_operators[:,3:4] / density_j
        # out = field_j * self.kernel_operators[:, 1:2] / density_j

        # grad_density = ((density_i+density_j)/2 - self.p[2])  * self.kernel_operators[:, 1:3] * self.p[1] / density_j  # / 1E7  # d_rho_x d_rho_y
        # velocity = self.kernel_operators[:, 0:1] * torch.sum(d_pos_j**2, dim=1)[:,None] / density_j
        # grad_velocity = self.kernel_operators[:, 1:3] * torch.sum(d_pos_j**2, dim=1)[:,None].repeat(1,2) / density_j.repeat(1,2)
        # out = torch.cat((grad_density, velocity, grad_velocity), dim = 1) # d_rho_x d_rho_y, velocity
        # out = field_j * self.kernel_operators[:, 1:2] / density_j
        # 

