import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import *
from matplotlib import pyplot as plt
import matplotlib


class PDE_F(pyg.nn.MessagePassing):

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
        self.p = p
        self.dimension = dimension
        self.delta_t = delta_t
        self.max_radius = max_radius
        self.bc_dpos = bc_dpos

        self.kernel_var = self.max_radius ** 2


    def forward(self, data, continuous_field=False, continuous_field_size=None):

        x, edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        particle_type = x[:, 1 + 2*self.dimension].long()
        parameters = self.p[particle_type, :]
        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        mass = self.p[particle_type, -1:]

        pos = pos + 1 * d_pos * self.delta_t

        field = x[:, 2*self.dimension+2: 2*self.dimension+3]

        if continuous_field:
            self.mode = 'kernel'
            previous_density = self.density
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, parameters=parameters, density=torch.zeros_like(x[:, 0:1]), mass=mass)

            self.density = self.density[:,0:1]

            density = torch.zeros_like(x[:, 0:1])
            density[continuous_field_size[0]:] = previous_density
            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, parameters=parameters, density=density, mass=mass)
        else:
            self.mode = 'kernel'
            self.density = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, parameters=parameters, density=torch.zeros_like(x[:, 0:1]), mass=mass)

            fig = plt.figure()
            plt.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), c=to_numpy(self.density[:, 2]), s=2, cmap='viridis', vmin=0, vmax=1E6)
            plt.colorbar()
            plt.savefig('density.tif')
            plt.close()

            self.density = self.density[:, 0:1]

            self.mode = 'message_passing'
            out = self.propagate(edge_index=edge_index, pos=pos, d_pos=d_pos, field=field, parameters=parameters, density=self.density, mass=mass)

        out[:, 1] = out[:, 1] + torch.ones_like(out[:, 1]) * parameters[:,0] * 9.8

        return out[:,0:2]


    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, field_i, field_j, parameters_i, density_i, density_j, mass_i, mass_j):

        delta_pos = self.bc_dpos(pos_j - pos_i)
        self.delta_pos = delta_pos

        if self.mode == 'kernel':
            mgrid = delta_pos.clone().detach()
            mgrid.requires_grad = True
            
            Gaussian_kernel = torch.exp(-4*(mgrid[:, 0] ** 2 + mgrid[:, 1] ** 2) / self.kernel_var)[:, None] / 2

            dist = torch.sqrt(torch.sum(mgrid ** 2, dim=1))
            triangle_kernel = ((self.max_radius - dist)**2 / self.kernel_var)[:, None] / 1.309

            grad_Gaussian_kernel = density_gradient(Gaussian_kernel, mgrid)
            grad_triangle_kernel = density_gradient(triangle_kernel, mgrid)
            grad_triangle_kernel = torch.where(torch.isnan(grad_triangle_kernel), torch.zeros_like(grad_triangle_kernel), grad_triangle_kernel)
            laplacian_kernel = density_laplace(Gaussian_kernel, mgrid)

            self.kernel_operators = dict()
            self.kernel_operators['Gaussian'] = Gaussian_kernel
            self.kernel_operators['grad_Gaussian'] = grad_Gaussian_kernel
            self.kernel_operators['grad_triangle'] = grad_triangle_kernel
            self.kernel_operators['laplacian'] = laplacian_kernel

            density = Gaussian_kernel * mass_j

            density = torch.cat((density, dist[:, None], torch.ones_like(dist[:, None])), dim=1)

            return density

            # fig=plt.figure()
            # plt.scatter(mgrid[:, 0].detach().cpu().numpy(), mgrid[:, 1].detach().cpu().numpy(), c=grad_triangle_kernel[:,0].detach().cpu().numpy(), cmap='viridis')
            # plt.show()

        elif self.mode == 'message_passing':

            pressure_force = mass_j * parameters_i[:,1:2] * torch.relu((density_i+density_j)/2 - parameters_i[:,2:3]/2) * self.kernel_operators['grad_triangle'] / density_j.repeat(1,2)
            viscosity_force = mass_j * parameters_i[:,3:4] * (d_pos_j-d_pos_i) * self.kernel_operators['Gaussian'].repeat(1,2) / density_j.repeat(1,2)
            convection_force_x = d_pos_i[:,0:1] * self.kernel_operators['Gaussian'][0:1] *  d_pos_j[:,0:1] + d_pos_i[:,1:2] * self.kernel_operators['Gaussian'][1:2] *  d_pos_j[:,0:1] / density_j
            convection_force_y = d_pos_i[:, 0:1] * self.kernel_operators['Gaussian'][0:1] * d_pos_j[:, 1:2] + d_pos_i[:,1:2] * self.kernel_operators['Gaussian'][0:1] * d_pos_j[:, 1:2] /density_j
            convection_force = convection_force_x + convection_force_y

            force = pressure_force + viscosity_force - convection_force / 100
            out = force / density_i

            return out

            # velocity = self.kernel_operators['Gaussian'] * torch.sum(d_pos_j ** 2, dim=1)[:, None] / density_j





    # viscosity_force = self.p[3] * d_pos_j * self.kernel_operators['laplacian'] / density_j.repeat(1,2)
    # convection_force_x = d_pos_i[:,0:1] * self.kernel_operators['Gaussian'][0:1] *  d_pos_j[:,0:1] + d_pos_i[:,1:2] * self.kernel_operators['Gaussian'][1:2] *  d_pos_j[:,0:1] / density_j
    # convection_force_y = d_pos_i[:, 0:1] * self.kernel_operators['Gaussian'][0:1] * d_pos_j[:, 1:2] + d_pos_i[:,1:2] * elf.kernel_operators['Gaussian'][0:1] * d_pos_j[:, 1:2] /density_j
    # convection_force = - self.p[4] * torch.cat((convection_force_x, convection_force_y), dim=1)
