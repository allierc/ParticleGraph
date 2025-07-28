
from ParticleGraph.generators import *
from ParticleGraph.utils import *
from time import sleep
from scipy.spatial import Delaunay
from tifffile import imread, imwrite as imsave
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange
from torch_geometric.utils import dense_to_sparse
from scipy import stats
import seaborn as sns



def choose_model(config=[], W=[], device=[]):
    particle_model_name = config.graph_model.particle_model_name
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    n_particles = config.simulation.n_particles
    delta_t = config.simulation.delta_t
    n_particle_types = config.simulation.n_particle_types
    short_term_plasticity_mode = config.simulation.short_term_plasticity_mode

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius

    params = config.simulation.params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    # create GNN depending in type specified in config file
    match particle_model_name:
        case 'PDE_A' | 'PDE_ParticleField_A' | 'PDE_Cell_A' :
            if config.simulation.non_discrete_level>0:
                p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
                pp=[]
                n_particle_types = len(params)
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
                for n in range(n_particle_types):
                    if n==0:
                        pp=p[n].repeat(n_particles//n_particle_types,1)
                    else:
                        pp=torch.cat((pp,p[n].repeat(n_particles//n_particle_types,1)),0)
                p=pp.clone().detach()
                p=p+torch.randn(n_particles,4,device=device) * config.simulation.non_discrete_level
            sigma = config.simulation.sigma
            p = p if n_particle_types == 1 else torch.squeeze(p)
            func_p = config.simulation.func_params
            embedding_step = config.simulation.n_frames // 100
            model = PDE_A(aggr_type=aggr_type, p=p, func_p = func_p, sigma=sigma, bc_dpos=bc_dpos, dimension=dimension, embedding_step=embedding_step)
        case 'PDE_B' | 'PDE_ParticleField_B' | 'PDE_Cell_B' | 'PDE_Cell_B_area':  # comprised between 10 and 50
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_B_mass':
            final_cell_mass = torch.tensor(config.simulation.final_cell_mass, device=device)
            model = PDE_B_mass(aggr_type=aggr_type, p=p, final_mass = final_cell_mass, bc_dpos=bc_dpos)
        case 'PDE_B_bis':
            model = PDE_B_bis(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos)
        case 'PDE_G':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_G(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_GS':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_GS(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_E':
            model = PDE_E(aggr_type=aggr_type, p=p,
                          clamp=config.training.clamp, pred_limit=config.training.pred_limit,
                          prediction=config.graph_model.prediction, bc_dpos=bc_dpos)

        case 'PDE_F' |'PDE_F_A' | 'PDE_F_B' :
            model = PDE_F(aggr_type=aggr_type, p=torch.tensor(params, dtype=torch.float32, device=device), bc_dpos=bc_dpos,
                          dimension=dimension, delta_t=delta_t, max_radius=max_radius, field_type=config.graph_model.field_type)
        case 'PDE_K':
            p = params
            edges = np.random.choice(p[0], size=(n_particles, n_particles), p=p[1])
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)
            connection_matrix = torch.tensor(edges, dtype=torch.float32, device=device)
            model = PDE_K(aggr_type=aggr_type, connection_matrix=connection_matrix, bc_dpos=bc_dpos)

        case 'PDE_O':
            model = PDE_O(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, beta=config.simulation.beta)
        case 'Maze':
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos)
        case _:
            model = PDE_Z(device=device)


    match config.simulation.phi:
        case 'tanh':
            phi=torch.tanh
        case 'relu':
            phi=torch.relu
        case 'sigmoid':
            phi=torch.sigmoid
        case _:
            phi=torch.sigmoid


    match model_signal_name:
        case 'PDE_N2':
            model = PDE_N2(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N3':
            model = PDE_N3(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N4':
            model = PDE_N4(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N5':
            model = PDE_N5(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N6':
            model = PDE_N6(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)
        case 'PDE_N7':
            model = PDE_N7(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)


    return model, bc_pos, bc_dpos


def choose_mesh_model(config, X1_mesh, device):
    mesh_model_name = config.graph_model.mesh_model_name
    n_node_types = config.simulation.n_node_types
    aggr_type = config.graph_model.mesh_aggr_type
    _, bc_dpos = choose_boundary_values(config.simulation.boundary)

    params = config.simulation.params
    delta_t = config.simulation.delta_t
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius

    if mesh_model_name =='':
        mesh_model = []
    else:
        # c = initialize_random_values(n_node_types, device)
        # if not('pics' in config.simulation.node_coeff_map):
        #     for n in range(n_node_types):
        #         c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])

        if config.simulation.node_coeff_map !='' :
            i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
        else:
            i0 = imread(f'graphs_data/pattern_Null.tif')
        i0 = np.flipud(i0)
        values = i0[(to_numpy(X1_mesh[:, 1]) * 255).astype(int), (to_numpy(X1_mesh[:, 0]) * 255).astype(int)]
        values = values
        values = np.reshape(values,len(X1_mesh))
        values = torch.tensor(values, device=device, dtype=torch.float32)[:, None]


        match mesh_model_name:
            case 'RD_Gray_Scott_Mesh':
                mesh_model = RD_Gray_Scott(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'RD_FitzHugh_Nagumo_Mesh':
                mesh_model = RD_FitzHugh_Nagumo(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'RD_Mesh':
                mesh_model = RD_RPS(aggr_type=aggr_type, bc_dpos=bc_dpos, coeff=values)
            case 'DiffMesh' | 'WaveMesh':
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, bc_dpos=bc_dpos, coeff=values)
            case 'WaveSmoothParticle':
                mesh_model = PDE_S(aggr_type=aggr_type, bc_dpos=bc_dpos, p=torch.tensor(params, dtype=torch.float32, device=device),
                          dimension=dimension, delta_t=delta_t, max_radius=max_radius, field_type=config.graph_model.field_type)
            case 'Chemotaxism_Mesh':
                c = initialize_random_values(n_node_types, device)
                for n in range(n_node_types):
                    c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case 'PDE_O_Mesh':
                c = initialize_random_values(n_node_types, device)
                for n in range(n_node_types):
                    c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, c=torch.squeeze(c), bc_dpos=bc_dpos)
            case _:
                mesh_model = PDE_Z(device=device)




    return mesh_model


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_particles(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_frames = config.simulation.n_frames
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if 'PDE_F' in config.graph_model.particle_model_name:
        pos = torch.rand(n_particles, dimension, device=device)
        if simulation_config.pos_init == 'square':
            pos = pos * 0.5 + 0.25
    elif (simulation_config.boundary == 'periodic') | ('PDE_K' in config.graph_model.particle_model_name):
        pos = torch.rand(n_particles, dimension, device=device)
        if n_particles <= 10:
            if 'PDE_K' in config.graph_model.particle_model_name:
                pos = pos * 0.5 + 0.25
            else:
                pos = pos * 0.1 + 0.45
        elif n_particles<=100:
            if 'PDE_K' in config.graph_model.particle_model_name:
                pos = pos * 0.4 + 0.2
            else:
                pos = pos * 0.2 + 0.4
        elif n_particles<=500:
            pos = pos * 0.5 + 0.25
    else:
        pos = torch.randn(n_particles, dimension, device=device) * 0.5
    dpos = dpos_init * torch.randn((n_particles, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(n_particles / n_particle_types), device=device)
    for n in range(1, n_particle_types):
        type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
    if type.shape[0] < n_particles:
        type = torch.cat((type, n * torch.ones(n_particles - type.shape[0], device=device)), 0)
    if (simulation_config.params == 'continuous') | (config.simulation.non_discrete_level > 0):  # TODO: params is a list[list[float]]; this can never happen?
        type = torch.tensor(np.arange(n_particles), device=device)

    if simulation_config.bounce:
        n_wall_particles = n_particles // n_particle_types
        n_particles_wall = n_wall_particles // 4
        wall_pos = torch.linspace(0.1, 0.9, n_particles_wall, device=device)
        wall0 = torch.zeros(n_particles_wall, 2, device=device)
        wall0[:,0] = wall_pos
        wall0[:,1] = 0.1
        wall1 = torch.zeros(n_particles_wall, 2, device=device)
        wall1[:,0] = wall_pos
        wall1[:,1] = 0.9
        wall2 = torch.zeros(n_particles_wall, 2, device=device)
        wall2[:,0] = 0.1
        wall2[:,1] = wall_pos
        wall3 = torch.zeros(n_particles_wall, 2, device=device)
        wall3[:,0] = 0.9
        wall3[:,1] = wall_pos
        pos_ = torch.cat((wall0,wall1,wall2,wall3), dim=0)
        pos_ = pos_ + torch.randn((n_wall_particles,dimension), device=device) * 0.001

        dpos [0:n_wall_particles] = 0
        pos [0:n_wall_particles:] = pos_

    features = torch.cat((torch.randn((n_particles, 1), device=device) * 5 , 0.1 * torch.randn((n_particles, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_particles,1), device=device)

    if 'pattern' in scenario:
        i0 = imread(f'graphs_data/pattern_0.tif')
        type = np.round(i0[(to_numpy(pos[:, 0]) * 255).astype(int), (to_numpy(pos[:, 1]) * 255).astype(int)] / 255 * n_particle_types-1).astype(int)
        type = torch.tensor(type, device=device)
        type = type[:, None]
    if 'uniform' in scenario :
        type = torch.ones(n_particles, device=device) * int(scenario.split()[-1])
        type =  type[:, None]
    if 'stripes' in scenario:
        l = n_particles//n_particle_types
        for n in range(n_particle_types):
            index = np.arange(n*l, (n+1)*l)
            pos[index, 1:2] = torch.rand(l, 1, device=device) * (1/n_particle_types) + n/n_particle_types

    return pos, dpos, type, features, age, particle_id


def init_neurons(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_frames = config.simulation.n_frames
    n_neurons = simulation_config.n_neurons * ratio
    n_neuron_types = simulation_config.n_neuron_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init


    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    pos = pos[perm]

    dpos = dpos_init * torch.randn((n_neurons, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))

    type = torch.zeros(int(n_neurons / n_neuron_types), device=device)

    for n in range(1, n_neuron_types):
        type = torch.cat((type, n * torch.ones(int(n_neurons / n_neuron_types), device=device)), 0)
    if type.shape[0] < n_neurons:
        type = torch.cat((type, n * torch.ones(n_neurons - type.shape[0], device=device)), 0)

    if (config.graph_model.signal_model_name == 'PDE_N6') | (config.graph_model.signal_model_name == 'PDE_N7'):
        features = torch.cat((torch.rand((n_neurons, 1), device=device), 0.1 * torch.randn((n_neurons, 1), device=device),
                              torch.ones((n_neurons, 1), device=device), torch.zeros((n_neurons, 1), device=device)), 1)
    elif 'excitation_single' in config.graph_model.field_type:
        features = torch.zeros((n_neurons, 2), device=device)
    else:
        features = torch.cat((torch.randn((n_neurons, 1), device=device) * 5 , 0.1 * torch.randn((n_neurons, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_neurons, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_neurons,1), device=device)

    return pos, dpos, type, features, age, particle_id


def init_MPM_shapes(
        geometry='cubes',  # 'cubes', 'discs', 'stars', 'letters'
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_particle_types=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        device='cpu'
):
    torch.manual_seed(seed)

    p_vol = (dx * 0.5) ** 2

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    v = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    C = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    F = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1, -1)
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    GM = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)
    GP = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Determine grid layout and spacing
    if n_shapes == 3:
        shape_row = group_indices
        shape_col = torch.zeros_like(group_indices)
        size, spacing, start_x, start_y = 0.1, 0.32, 0.3, 0.15
    elif n_shapes == 9:
        shape_row = group_indices // 3
        shape_col = group_indices % 3
        size, spacing, start_x, start_y = 0.075, 0.25, 0.2, 0.375
    elif n_shapes == 16:
        shape_row = group_indices // 4
        shape_col = group_indices % 4
        size, spacing, start_x, start_y = 0.04, 0.2, 0.2, 0.2
    elif n_shapes == 25:
        shape_row = group_indices // 5
        shape_col = group_indices % 5
        size, spacing, start_x, start_y = 0.035, 0.16, 0.15, 0.2
    elif n_shapes == 36:
        shape_row = group_indices // 6
        shape_col = group_indices % 6
        size, spacing, start_x, start_y = 0.035, 0.13, 0.1, 0.14
    else:
        # General case: try to make a square grid
        grid_size = int(n_shapes ** 0.5)
        shape_row = group_indices // grid_size
        shape_col = group_indices % grid_size
        size, spacing = 0.4 / grid_size, 0.8 / grid_size
        start_x, start_y = 0.1 + size, 0.1 + size

    center_x = start_x + spacing * shape_col.float()
    center_y = start_y + spacing * shape_row.float()

    # Random rotation angles for each shape (except discs)
    shape_rotations = torch.rand(n_shapes, device=device) * 2 * torch.pi

    # Define letter shapes as line segments (relative to center, normalized to [-1,1])
    def get_letter_segments(letter):
        if letter == 'A':
            return [
                [[-0.5, -1], [0, 1]],  # Left diagonal
                [[0.5, -1], [0, 1]],  # Right diagonal
                [[-0.25, 0], [0.25, 0]]  # Crossbar
            ]
        elif letter == 'E':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Vertical line
                [[-0.5, 1], [0.5, 1]],  # Top horizontal
                [[-0.5, 0], [0.2, 0]],  # Middle horizontal
                [[-0.5, -1], [0.5, -1]]  # Bottom horizontal
            ]
        elif letter == 'F':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Vertical line
                [[-0.5, 1], [0.5, 1]],  # Top horizontal
                [[-0.5, 0], [0.2, 0]]  # Middle horizontal
            ]
        elif letter == 'H':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Left vertical
                [[0.5, -1], [0.5, 1]],  # Right vertical
                [[-0.5, 0], [0.5, 0]]  # Crossbar
            ]
        elif letter == 'I':
            return [
                [[-0.3, 1], [0.3, 1]],  # Top horizontal
                [[0, 1], [0, -1]],  # Vertical line
                [[-0.3, -1], [0.3, -1]]  # Bottom horizontal
            ]
        elif letter == 'L':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Vertical line
                [[-0.5, -1], [0.5, -1]]  # Bottom horizontal
            ]
        elif letter == 'T':
            return [
                [[-0.5, 1], [0.5, 1]],  # Top horizontal
                [[0, 1], [0, -1]]  # Vertical line
            ]
        else:  # Default to 'O' (circle-like)
            # Approximate circle with 8 line segments
            angles = torch.linspace(0, 2 * torch.pi, 9)
            segments = []
            for i in range(8):
                x1, y1 = 0.5 * torch.cos(angles[i]), 0.5 * torch.sin(angles[i])
                x2, y2 = 0.5 * torch.cos(angles[i + 1]), 0.5 * torch.sin(angles[i + 1])
                segments.append([[x1.item(), y1.item()], [x2.item(), y2.item()]])
            return segments

    # Generate particles based on geometry
    if geometry == 'cubes':
        # Generate cube particles relative to center
        rel_x = torch.rand(n_particles, device=device) * (size * 2) - size
        rel_y = torch.rand(n_particles, device=device) * (size * 2) - size

        # Apply rotation to each cube
        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = start_idx + group_size

            angle = shape_rotations[shape_idx]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)

            # Rotate relative positions
            rotated_x = rel_x[start_idx:end_idx] * cos_a - rel_y[start_idx:end_idx] * sin_a
            rotated_y = rel_x[start_idx:end_idx] * sin_a + rel_y[start_idx:end_idx] * cos_a

            # Add center position
            x[start_idx:end_idx, 0] = center_x[start_idx] + rotated_x
            x[start_idx:end_idx, 1] = center_y[start_idx] + rotated_y

    elif geometry == 'discs':
        # Use the better circular generation
        outer_radius = size

        particles_per_shape = group_size
        valid_particles = []

        for shape_idx in range(n_shapes):
            shape_particles = []

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]

            # Generate particles in circular pattern
            for _ in range(particles_per_shape):
                r_test = torch.rand(1, device=device).sqrt() * outer_radius
                theta_test = torch.rand(1, device=device) * 2 * torch.pi

                px = shape_center_x + r_test * torch.cos(theta_test)
                py = shape_center_y + r_test * torch.sin(theta_test)
                shape_particles.append([px.item(), py.item()])

            valid_particles.extend(shape_particles)

        disc_positions = torch.tensor(valid_particles[:n_particles], device=device)
        x[:, 0] = disc_positions[:, 0]
        x[:, 1] = disc_positions[:, 1]

    elif geometry == 'stars':
        outer_radius = size
        inner_radius = outer_radius * 0.4
        n_points = 5

        particles_per_shape = group_size
        valid_particles = []

        for shape_idx in range(n_shapes):
            shape_particles = []

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]

            # Apply rotation to star
            rotation_angle = shape_rotations[shape_idx]

            # Create 5-pointed star vertices with rotation
            star_angles = torch.linspace(0, 2 * torch.pi, n_points * 2 + 1, device=device)[:-1] + rotation_angle
            star_radii = torch.zeros_like(star_angles)
            star_radii[::2] = outer_radius  # Outer points
            star_radii[1::2] = inner_radius  # Inner points

            # Create star vertices for this shape
            star_x = shape_center_x + star_radii * torch.cos(star_angles)
            star_y = shape_center_y + star_radii * torch.sin(star_angles)

            # Fill star by generating particles in triangular sectors
            particles_per_triangle = particles_per_shape // n_points

            for i in range(n_points):
                # Each star triangle: center -> outer point -> inner point -> next outer point
                p1_x, p1_y = shape_center_x, shape_center_y  # Center
                p2_x, p2_y = star_x[i * 2], star_y[i * 2]  # Outer point
                p3_x, p3_y = star_x[(i * 2 + 1) % (n_points * 2)], star_y[(i * 2 + 1) % (n_points * 2)]  # Inner point
                p4_x, p4_y = star_x[(i * 2 + 2) % (n_points * 2)], star_y[
                    (i * 2 + 2) % (n_points * 2)]  # Next outer point

                # Fill triangle (center, outer, inner)
                for _ in range(particles_per_triangle // 2):
                    r1, r2 = torch.rand(2, device=device)
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2

                    px = p1_x + r1 * (p2_x - p1_x) + r2 * (p3_x - p1_x)
                    py = p1_y + r1 * (p2_y - p1_y) + r2 * (p3_y - p1_y)
                    shape_particles.append([px.item(), py.item()])

                # Fill triangle (center, inner, next outer)
                for _ in range(particles_per_triangle // 2):
                    r1, r2 = torch.rand(2, device=device)
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2

                    px = p1_x + r1 * (p3_x - p1_x) + r2 * (p4_x - p1_x)
                    py = p1_y + r1 * (p3_y - p1_y) + r2 * (p4_y - p1_y)
                    shape_particles.append([px.item(), py.item()])

            # Fill any remaining particles
            while len(shape_particles) < particles_per_shape:
                r_fill = torch.rand(1, device=device).sqrt() * inner_radius * 0.5
                theta_fill = torch.rand(1, device=device) * 2 * torch.pi
                px = shape_center_x + r_fill * torch.cos(theta_fill)
                py = shape_center_y + r_fill * torch.sin(theta_fill)
                shape_particles.append([px.item(), py.item()])

            valid_particles.extend(shape_particles[:particles_per_shape])

        star_positions = torch.tensor(valid_particles[:n_particles], device=device)
        x[:, 0] = star_positions[:, 0]
        x[:, 1] = star_positions[:, 1]

    elif geometry == 'letters':
        # Available letters
        letters = ['A', 'E', 'F', 'H', 'I', 'L', 'T', 'O']

        particles_per_shape = group_size
        valid_particles = []

        for shape_idx in range(n_shapes):
            shape_particles = []

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]

            # Choose random letter for this shape
            letter = letters[torch.randint(0, len(letters), (1,)).item()]
            segments = get_letter_segments(letter)

            # Apply rotation
            rotation_angle = shape_rotations[shape_idx]
            cos_a, sin_a = torch.cos(rotation_angle), torch.sin(rotation_angle)

            # Sample particles along letter segments
            particles_per_segment = particles_per_shape // len(segments)

            for segment in segments:
                x1, y1 = segment[0]
                x2, y2 = segment[1]

                for _ in range(particles_per_segment):
                    # Sample point along line segment
                    t = torch.rand(1, device=device)
                    rel_x = x1 + t * (x2 - x1)
                    rel_y = y1 + t * (y2 - y1)

                    # Scale by size
                    rel_x *= size
                    rel_y *= size

                    # Apply rotation
                    rotated_x = rel_x * cos_a - rel_y * sin_a
                    rotated_y = rel_x * sin_a + rel_y * cos_a

                    # Add center position
                    px = shape_center_x + rotated_x
                    py = shape_center_y + rotated_y
                    shape_particles.append([px.item(), py.item()])

            # Fill any remaining particles near center
            while len(shape_particles) < particles_per_shape:
                r_fill = torch.rand(1, device=device) * size * 0.1
                theta_fill = torch.rand(1, device=device) * 2 * torch.pi
                px = shape_center_x + r_fill * torch.cos(theta_fill)
                py = shape_center_y + r_fill * torch.sin(theta_fill)
                shape_particles.append([px.item(), py.item()])

            valid_particles.extend(shape_particles[:particles_per_shape])

        letter_positions = torch.tensor(valid_particles[:n_particles], device=device)
        x[:, 0] = letter_positions[:, 0]
        x[:, 1] = letter_positions[:, 1]

    # Random materials for each shape
    if n_particle_types> 1:
        shape_materials = torch.randperm(n_shapes, device=device) % n_particle_types
        T = shape_materials[group_indices].unsqueeze(1).int()

    # Calculate mass based on material type
    # Material 0: water (density = 1.0)
    # Material 1: jelly (density = 0.5, twice lighter than water)
    # Material 2: snow (density = 0.25, four times lighter than water)
    material_densities = torch.tensor(rho_list, device=device)
    particle_densities = material_densities[T.squeeze()]
    M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device) * particle_densities.unsqueeze(1)

    # Random velocity per shape
    shape_velocities = (torch.rand(n_shapes, 2, device=device) - 0.5) * 4.0
    v = shape_velocities[group_indices]

    # Object ID for each particle
    ID = group_indices.unsqueeze(1).int()
    id_permutation = torch.randperm(n_shapes, device=device)
    ID = id_permutation[ID.squeeze()].unsqueeze(1)


    return N, x, v, C, F, T, Jp, M, S, ID


def init_MPM_3D_shapes(
        geometry='cubes',  # 'cubes', 'spheres', 'stars', 'letters'
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_particle_types=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        device='cpu'
):
    torch.manual_seed(seed)

    # 3D volume instead of 2D area
    p_vol = (dx * 0.5) ** 3

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 3), dtype=torch.float32, device=device)  # 3D positions
    v = torch.zeros((n_particles, 3), dtype=torch.float32, device=device)  # 3D velocities
    C = torch.zeros((n_particles, 3, 3), dtype=torch.float32, device=device)  # 3x3 affine matrix
    F = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1,
                                                                             -1)  # 3x3 deformation gradient
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 3, 3), dtype=torch.float32, device=device)  # 3x3 stress tensor
    GM = torch.zeros((n_grid, n_grid, n_grid), dtype=torch.float32, device=device)  # 3D grid mass
    GP = torch.zeros((n_grid, n_grid, n_grid), dtype=torch.float32, device=device)  # 3D grid momentum

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Determine 3D grid layout and spacing
    if n_shapes == 3:
        # Linear arrangement along x-axis
        shape_row = torch.zeros_like(group_indices)
        shape_col = torch.zeros_like(group_indices)
        shape_depth = group_indices
        size, spacing_x, spacing_y, spacing_z = 0.1, 0.32, 0.0, 0.32
        start_x, start_y, start_z = 0.3, 0.4, 0.15
    elif n_shapes == 8:
        # 2x2x2 cube arrangement
        shape_depth = group_indices // 4
        temp = group_indices % 4
        shape_row = temp // 2
        shape_col = temp % 2
        size, spacing_x, spacing_y, spacing_z = 0.08, 0.3, 0.3, 0.3
        start_x, start_y, start_z = 0.25, 0.25, 0.25
    elif n_shapes == 9:
        # 3x3x1 arrangement
        shape_depth = torch.zeros_like(group_indices)
        shape_row = group_indices // 3
        shape_col = group_indices % 3
        size, spacing_x, spacing_y, spacing_z = 0.075, 0.25, 0.25, 0.0
        start_x, start_y, start_z = 0.2, 0.2, 0.4
    elif n_shapes == 16:
        # 4x4x1 arrangement
        shape_depth = torch.zeros_like(group_indices)
        shape_row = group_indices // 4
        shape_col = group_indices % 4
        size, spacing_x, spacing_y, spacing_z = 0.04, 0.2, 0.2, 0.0
        start_x, start_y, start_z = 0.2, 0.2, 0.4
    elif n_shapes == 25:
        # 5x5x1 arrangement
        shape_depth = torch.zeros_like(group_indices)
        shape_row = group_indices // 5
        shape_col = group_indices % 5
        size, spacing_x, spacing_y, spacing_z = 0.035, 0.16, 0.16, 0.0
        start_x, start_y, start_z = 0.15, 0.15, 0.4
    elif n_shapes == 27:
        # 3x3x3 cube arrangement
        shape_depth = group_indices // 9
        temp = group_indices % 9
        shape_row = temp // 3
        shape_col = temp % 3
        size, spacing_x, spacing_y, spacing_z = 0.06, 0.25, 0.25, 0.25
        start_x, start_y, start_z = 0.2, 0.2, 0.2
    else:
        # General case: try to make a cubic grid
        grid_size = int(round(n_shapes ** (1 / 3)))
        if grid_size ** 3 < n_shapes:
            grid_size += 1

        shape_depth = group_indices // (grid_size * grid_size)
        temp = group_indices % (grid_size * grid_size)
        shape_row = temp // grid_size
        shape_col = temp % grid_size

        size = 0.8 / (grid_size + 1)
        spacing_x = spacing_y = spacing_z = 0.8 / grid_size
        start_x = start_y = start_z = 0.1

    # Calculate center positions for each shape
    center_x = start_x + shape_col.float() * spacing_x
    center_y = start_y + shape_row.float() * spacing_y
    center_z = start_z + shape_depth.float() * spacing_z

    # Generate particles within each shape
    if geometry == 'cubes':
        # Generate particles in cubic volumes
        particles_per_dim = int(round((group_size) ** (1 / 3)))
        if particles_per_dim ** 3 < group_size:
            particles_per_dim += 1

        for i in range(n_particles):
            group_idx = i // group_size
            local_idx = i % group_size

            # 3D indexing within cube
            z_idx = local_idx // (particles_per_dim * particles_per_dim)
            temp = local_idx % (particles_per_dim * particles_per_dim)
            y_idx = temp // particles_per_dim
            x_idx = temp % particles_per_dim

            # Normalize to [-0.5, 0.5] range then scale and translate
            local_x = (x_idx / max(particles_per_dim - 1, 1) - 0.5) * size
            local_y = (y_idx / max(particles_per_dim - 1, 1) - 0.5) * size
            local_z = (z_idx / max(particles_per_dim - 1, 1) - 0.5) * size

            x[i, 0] = center_x[i] + local_x
            x[i, 1] = center_y[i] + local_y
            x[i, 2] = center_z[i] + local_z

    elif geometry == 'spheres':
        # Generate particles in spherical volumes
        for i in range(n_particles):
            group_idx = i // group_size

            # Generate random point in unit sphere using rejection sampling
            while True:
                rand_x = torch.rand(1, device=device) * 2 - 1
                rand_y = torch.rand(1, device=device) * 2 - 1
                rand_z = torch.rand(1, device=device) * 2 - 1

                if rand_x ** 2 + rand_y ** 2 + rand_z ** 2 <= 1.0:
                    break

            # Scale by size and translate to center
            x[i, 0] = center_x[i] + rand_x * size * 0.5
            x[i, 1] = center_y[i] + rand_y * size * 0.5
            x[i, 2] = center_z[i] + rand_z * size * 0.5

    else:  # Default to cubes
        # Same as cubes case
        particles_per_dim = int(round((group_size) ** (1 / 3)))
        if particles_per_dim ** 3 < group_size:
            particles_per_dim += 1

        for i in range(n_particles):
            group_idx = i // group_size
            local_idx = i % group_size

            z_idx = local_idx // (particles_per_dim * particles_per_dim)
            temp = local_idx % (particles_per_dim * particles_per_dim)
            y_idx = temp // particles_per_dim
            x_idx = temp % particles_per_dim

            local_x = (x_idx / max(particles_per_dim - 1, 1) - 0.5) * size
            local_y = (y_idx / max(particles_per_dim - 1, 1) - 0.5) * size
            local_z = (z_idx / max(particles_per_dim - 1, 1) - 0.5) * size

            x[i, 0] = center_x[i] + local_x
            x[i, 1] = center_y[i] + local_y
            x[i, 2] = center_z[i] + local_z

    # Assign material types
    for i in range(n_particles):
        group_idx = i // group_size
        T[i] = group_idx % n_particle_types  # Cycle through material types 0, 1, 2, ...

    # Generate masses based on density
    M = torch.ones((n_particles, 1), dtype=torch.float32, device=device) * p_vol
    for i in range(n_particles):
        group_idx = i // group_size
        if group_idx < len(rho_list):
            M[i] = M[i] * rho_list[group_idx]
        else:
            # Use the last density if we run out of densities
            M[i] = M[i] * rho_list[-1] if len(rho_list) > 0 else M[i]

    # Generate shape IDs for tracking
    ID = torch.zeros((n_particles, 1), dtype=torch.int32, device=device)
    for i in range(n_particles):
        ID[i] = i // group_size

    return N, x, v, C, F, T, Jp, M, S, ID

def init_MPM_cells(
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        nucleus_ratio=0.6,  # nucleus radius / total radius
        device='cpu'
):
    torch.manual_seed(seed)

    p_vol = (dx * 0.5) ** 2

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    v = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    C = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    F = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1, -1)
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    GM = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)
    GP = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Determine grid layout and spacing
    if n_shapes == 3:
        shape_row = group_indices
        shape_col = torch.zeros_like(group_indices)
        size, spacing, start_x, start_y = 0.1, 0.32, 0.3, 0.15
    elif n_shapes == 9:
        shape_row = group_indices // 3
        shape_col = group_indices % 3
        size, spacing, start_x, start_y = 0.075, 0.25, 0.2, 0.375
    elif n_shapes == 16:
        shape_row = group_indices // 4
        shape_col = group_indices % 4
        size, spacing, start_x, start_y = 0.04, 0.2, 0.2, 0.2
    elif n_shapes == 25:
        shape_row = group_indices // 5
        shape_col = group_indices % 5
        size, spacing, start_x, start_y = 0.035, 0.16, 0.15, 0.2
    elif n_shapes == 36:
        shape_row = group_indices // 6
        shape_col = group_indices % 6
        size, spacing, start_x, start_y = 0.035, 0.13, 0.1, 0.14
    else:
        # General case: try to make a square grid
        grid_size = int(n_shapes ** 0.5)
        shape_row = group_indices // grid_size
        shape_col = group_indices % grid_size
        size, spacing = 0.4 / grid_size, 0.8 / grid_size
        start_x, start_y = 0.1 + size, 0.1 + size

    center_x = start_x + spacing * shape_col.float()
    center_y = start_y + spacing * shape_row.float()

    # Generate cell particles (discs with nucleus and membrane)
    outer_radius = size
    nucleus_radius = outer_radius * nucleus_ratio

    particles_per_shape = group_size
    valid_particles = []
    particle_materials = []

    # Calculate particles distribution: nucleus area vs membrane area
    nucleus_area = torch.pi * nucleus_radius ** 2
    membrane_area = torch.pi * (outer_radius ** 2 - nucleus_radius ** 2)
    total_area = nucleus_area + membrane_area

    particles_nucleus = int(particles_per_shape * nucleus_area / total_area)
    particles_membrane = particles_per_shape - particles_nucleus

    for shape_idx in range(n_shapes):
        shape_particles = []
        shape_materials = []

        shape_center_x = center_x[shape_idx * group_size]
        shape_center_y = center_y[shape_idx * group_size]

        # Generate nucleus particles (material 0 - liquid)
        for _ in range(particles_nucleus):
            r_test = torch.rand(1, device=device).sqrt() * nucleus_radius
            theta_test = torch.rand(1, device=device) * 2 * torch.pi

            px = shape_center_x + r_test * torch.cos(theta_test)
            py = shape_center_y + r_test * torch.sin(theta_test)
            shape_particles.append([px.item(), py.item()])
            shape_materials.append(0)  # Material 0 for nucleus

        # Generate membrane particles (material 1 - jelly)
        for _ in range(particles_membrane):
            # Sample in annular region between nucleus_radius and outer_radius
            r_min_sq = nucleus_radius ** 2
            r_max_sq = outer_radius ** 2
            r_test = torch.sqrt(torch.rand(1, device=device) * (r_max_sq - r_min_sq) + r_min_sq)
            theta_test = torch.rand(1, device=device) * 2 * torch.pi

            px = shape_center_x + r_test * torch.cos(theta_test)
            py = shape_center_y + r_test * torch.sin(theta_test)
            shape_particles.append([px.item(), py.item()])
            shape_materials.append(1)  # Material 1 for membrane

        valid_particles.extend(shape_particles)
        particle_materials.extend(shape_materials)

    cell_positions = torch.tensor(valid_particles[:n_particles], device=device)
    x[:, 0] = cell_positions[:, 0]
    x[:, 1] = cell_positions[:, 1]

    # Set materials based on nucleus/membrane assignment
    T = torch.tensor(particle_materials[:n_particles], device=device).unsqueeze(1).int()

    # Calculate mass based on material type
    # Material 0: liquid (nucleus)
    # Material 1: jelly (membrane)
    material_densities = torch.tensor(rho_list, device=device)
    particle_densities = material_densities[T.squeeze()]
    M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device) * particle_densities.unsqueeze(1)

    # Random velocity per shape
    shape_velocities = (torch.rand(n_shapes, 2, device=device) - 0.5) * 4.0
    v = shape_velocities[group_indices]

    # Object ID for each particle
    ID = group_indices.unsqueeze(1).int()
    id_permutation = torch.randperm(n_shapes, device=device)
    ID = id_permutation[ID.squeeze()].unsqueeze(1)

    return N, x, v, C, F, T, Jp, M, S, ID


def get_index(n_particles, n_particle_types):
    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    return index_particles


def get_time_series(x_list, cell_id, feature):

    match feature:
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case 'type' | 'state':
            feature = 5
        case 'age':
            feature = 8
        case 'mass':
            feature = 10

        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))

    return to_numpy(torch.stack(time_series))


def init_mesh(config, device):

    simulation_config = config.simulation
    model_config = config.graph_model

    n_nodes = simulation_config.n_nodes
    n_particles = simulation_config.n_particles
    node_value_map = simulation_config.node_value_map
    field_grid = model_config.field_grid
    max_radius = simulation_config.max_radius

    n_nodes_per_axis = int(np.sqrt(n_nodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos_mesh = torch.zeros((n_nodes, 2), device=device)
    pos_mesh[0:n_nodes, 0:1] = x_mesh[0:n_nodes]
    pos_mesh[0:n_nodes, 1:2] = y_mesh[0:n_nodes]

    i0 = imread(f'graphs_data/{node_value_map}')
    if len(i0.shape) == 2:
        # i0 = i0[0,:, :]
        i0 = np.flipud(i0)
        values = i0[(to_numpy(pos_mesh[:, 1]) * 255).astype(int), (to_numpy(pos_mesh[:, 0]) * 255).astype(int)]

    mask_mesh = (x_mesh > torch.min(x_mesh) + 0.02) & (x_mesh < torch.max(x_mesh) - 0.02) & (y_mesh > torch.min(y_mesh) + 0.02) & (y_mesh < torch.max(y_mesh) - 0.02)

    if 'grid' in field_grid:
        pos_mesh = pos_mesh
    else:
        if 'pattern_Null.tif' in simulation_config.node_value_map:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 24
        else:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            node_value = torch.zeros((n_nodes, 2), device=device)
            node_value[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            node_value[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            node_value = torch.zeros((n_nodes, 2), device=device) + torch.rand((n_nodes, 2), device=device) * 0.1
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' :
            node_value = torch.rand((n_nodes, 3), device=device)
            s = torch.sum(node_value, dim=1)
            for k in range(3):
                node_value[:, k] = node_value[:, k] / s
        case 'DiffMesh' | 'WaveMesh' | 'Particle_Mesh_A' | 'Particle_Mesh_B' | 'WaveSmoothParticle':
            node_value = torch.zeros((n_nodes, 2), device=device)
            node_value[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            node_value = torch.zeros((n_particles, 5), device=device)
            node_value[0:n_particles, 0:1] = x_mesh[0:n_particles]
            node_value[0:n_particles, 1:2] = y_mesh[0:n_particles]
            node_value[0:n_particles, 2:3] = torch.randn(n_particles, 1, device=device) * 2 * np.pi  # theta
            node_value[0:n_particles, 3:4] = torch.ones(n_particles, 1, device=device) * np.pi / 200  # d_theta
            node_value[0:n_particles, 4:5] = node_value[0:n_particles, 3:4]  # d_theta0
            pos_mesh[:, 0] = node_value[:, 0] + (3 / 8) * mesh_size * torch.cos(node_value[:, 2])
            pos_mesh[:, 1] = node_value[:, 1] + (3 / 8) * mesh_size * torch.sin(node_value[:, 2])
        case '' :
            node_value = torch.zeros((n_nodes, 2), device=device)

    # i0 = imread(f'graphs_data/{node_type_map}')
    # values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
    # if np.max(values) > 0:
    #     values = np.round(values / np.max(values) * (simulation_config.n_node_types-1))
    # type_mesh = torch.tensor(values, device=device)
    # type_mesh = type_mesh[:, None]

    type_mesh = torch.zeros((n_nodes, 1), device=device)

    node_id_mesh = torch.arange(n_nodes, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((n_nodes, 2), device=device)

    x_mesh = torch.concatenate((node_id_mesh.clone().detach(), pos_mesh.clone().detach(), dpos_mesh.clone().detach(),
                                type_mesh.clone().detach(), node_value.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    print('removal of skinny faces ...')
    sleep(0.5)
    for k in trange(face.shape[0]):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)
    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    if (config.graph_model.particle_model_name == 'PDE_ParticleField_A')  | (config.graph_model.particle_model_name == 'PDE_ParticleField_B'):
        type_mesh = 0 * type_mesh

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)

    if 'Smooth' in config.graph_model.mesh_model_name:
        distance = torch.sum((pos_mesh[:, None, :] - pos_mesh[None, :, :]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= 0)).float() * 1
        mesh_data['edge_index'] = adj_t.nonzero().t().contiguous()


    return pos_mesh, dpos_mesh, type_mesh, node_value, a_mesh, node_id_mesh, mesh_data


def init_synapse_map(config, x, edge_attr_adjacency, device):

    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
    G = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED

        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,

        # Log
        verbose=True)

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=500)
    positions = np.array(list(positions.values()))
    X1 = torch.tensor(positions, dtype=torch.float32, device=device)
    X1 = X1 - torch.mean(X1, 0)

    torch.save(X1, f'./graphs_data/graphs_{dataset_name}/X1.pt')

    x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                           H1.clone().detach(), A1.clone().detach()), 1)

    # pos = nx.spring_layout(G, weight='weight', seed=42, k=1)
    # for k,p in pos.items():
    #     X1[k,:] = torch.tensor([p[0],p[1]], device=device)
    
    
def init_connectivity(connectivity_file, connectivity_distribution, connectivity_filling_factor, T1, n_particles, n_particle_types, dataset_name, device):

    if 'adjacency.pt' in connectivity_file:
        adjacency = torch.load(connectivity_file, map_location=device)

    elif 'mat' in connectivity_file:
        mat = scipy.io.loadmat(connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)

    elif 'zarr' in connectivity_file:
        print('loading zarr ...')
        dataset = xr.open_zarr(connectivity_file)
        trained_weights = dataset["trained"]  # alpha * sign * N
        print(f'weights {trained_weights.shape}')
        untrained_weights = dataset["untrained"]  # sign * N
        values = trained_weights[0:n_particles,0:n_particles]
        values = np.array(values)
        values = values / np.max(values)
        adjacency = torch.tensor(values, dtype=torch.float32, device=device)
        values=[]

    elif 'tif' in connectivity_file:
        adjacency = constructRandomMatrices(n_neurons=n_particles, density=1.0, connectivity_mask=f"./graphs_data/{connectivity_file}" ,device=device)
        n_particles = adjacency.shape[0]
        config.simulation.n_particles = n_particles

    elif 'values' in connectivity_file:
        parts = connectivity_file.split('_')
        w01 = float(parts[-2])
        w10 = float(parts[-1])
        adjacency =[[0, w01], [w10, 0]]
        adjacency = np.array(adjacency)
        adjacency = torch.tensor(adjacency, dtype=torch.float32, device=device)

    else:

        if 'Gaussian' in connectivity_distribution:
            adjacency = torch.randn((n_particles, n_particles), dtype=torch.float32, device=device)
            adjacency = adjacency / np.sqrt(n_particles)
            print(f"Gaussian   1/sqrt(N)  {1/np.sqrt(n_particles)}    std {torch.std(adjacency.flatten())}")

        elif 'Lorentz' in connectivity_distribution:

            s = np.random.standard_cauchy(n_particles**2)
            s[(s < -25) | (s > 25)] = 0

            if n_particles < 2000:
                s = s / n_particles**0.7
            elif n_particles <4000:
                s = s / n_particles**0.675
            elif n_particles < 8000:
                s = s / n_particles**0.67
            elif n_particles == 8000:
                s = s / n_particles**0.66
            elif n_particles > 8000:
                s = s / n_particles**0.5
            print(f"Lorentz   1/sqrt(N)  {1/np.sqrt(n_particles):0.3f}    std {np.std(s):0.3f}")

            adjacency = torch.tensor(s, dtype=torch.float32, device=device)
            adjacency = torch.reshape(adjacency, (n_particles, n_particles))

        elif 'uniform' in connectivity_distribution:
            adjacency = torch.rand((n_particles, n_particles), dtype=torch.float32, device=device)
            adjacency = adjacency - 0.5

        i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
        adjacency[i, i] = 0

    if connectivity_filling_factor != 1:
        mask = torch.rand(adjacency.shape) >  connectivity_filling_factor
        adjacency[mask] = 0
        mask = (adjacency != 0).float()
        # edge_index_, edge_attr_ = dense_to_sparse(adjacency)
        if n_particles>10000:
            edge_index = large_tensor_nonzero(mask)
            print (f'edge_index {edge_index.shape}')
        else:
            edge_index = mask.nonzero().t().contiguous()

    else:
        adj_matrix = torch.ones((n_particles)) - torch.eye(n_particles)
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        mask = (adj_matrix != 0).float()

    if 'structured' in connectivity_distribution:
        parts = connectivity_distribution.split('_')
        float_value1 = float(parts[-2])  # repartition pos/neg
        float_value2 = float(parts[-1])  # filling factor

        matrix_sign = torch.tensor(stats.bernoulli(float_value1).rvs(n_particle_types ** 2) * 2 - 1,
                                   dtype=torch.float32, device=device)
        matrix_sign = matrix_sign.reshape(n_particle_types, n_particle_types)

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_0.png', dpi=300)
        plt.close()

        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        adjacency = torch.abs(adjacency)
        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        sign_matrix = matrix_sign[xy_grid[..., 0], xy_grid[..., 1]]
        adjacency *= sign_matrix

        plt.imshow(to_numpy(sign_matrix))
        plt.savefig(f"graphs_data/{dataset_name}/large_connectivity_sign.tif", dpi=130)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_1.png', dpi=300)
        plt.close()

        flat_sign_matrix = sign_matrix.flatten()
        num_elements = len(flat_sign_matrix)
        num_ones = int(num_elements * float_value2)
        indices = np.random.choice(num_elements, num_ones, replace=False)
        flat_sign_matrix[:] = 0
        flat_sign_matrix[indices] = 1
        sign_matrix = flat_sign_matrix.reshape(sign_matrix.shape)

        adjacency *= sign_matrix

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_2.png', dpi=300)
        plt.close()

    edge_index = edge_index.to(device=device)

    return edge_index, adjacency, mask



