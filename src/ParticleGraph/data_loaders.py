"""
A collection of functions for loading data from various sources.
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import torch
from astropy.units import Unit
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline
from tqdm import trange

from ParticleGraph.TimeSeries import TimeSeries
from ParticleGraph.utils import *


def skip_to(file, start_line):
    with open(file) as f:
        pos = 0
        cur_line = f.readline()
        while cur_line != start_line:
            pos += 1
            cur_line = f.readline()

        return pos + 1


def load_solar_system(config, device=None, visualize=False, step=1000):
    # create output folder, empty it if bErase=True, copy files into it
    dataset_name = config.data_folder_name
    simulation_config = config.simulation
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_step = simulation_config.n_frames + 3
    n_frames = simulation_config.n_frames
    # Start = 1980 - 03 - 06
    # Stop = 2013 - 03 - 06
    # Step = 4(hours)

    object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'io',
                   'europa', 'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea', 'titan', 'hyperion', 'moon',
                   'phobos', 'deimos', 'charon']

    # matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(12, 12))

    all_data = []

    for id, object in enumerate(object_list):

        print(f'object: {object}')
        filename = os.path.join(dataset_name, f'{object}.txt')

        df = skip_to(filename, "$$SOE\n")
        data = pd.read_csv(filename, header=None, skiprows=df, nrows=n_step)
        x = data.iloc[:, 2:3].values
        y = data.iloc[:, 3:4].values
        z = data.iloc[:, 4:5].values

        # convert string to float
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        z = torch.tensor(z, dtype=torch.float32, device=device)
        vx = torch.zeros_like(x)
        vy = torch.zeros_like(y)
        vz = torch.zeros_like(z)
        vx[1:] = (x[1:] - x[:-1]) / simulation_config.delta_t
        vy[1:] = (y[1:] - y[:-1]) / simulation_config.delta_t
        vz[1:] = (z[1:] - z[:-1]) / simulation_config.delta_t
        ax = torch.zeros_like(x)
        ay = torch.zeros_like(y)
        az = torch.zeros_like(z)
        ax[2:] = (vx[2:] - vx[1:-1]) / simulation_config.delta_t
        ay[2:] = (vy[2:] - vy[1:-1]) / simulation_config.delta_t
        az[2:] = (vz[2:] - vz[1:-1]) / simulation_config.delta_t

        object_data = torch.cat((torch.ones_like(x[:, None]) * id, x[:, None], y[:, None], z[:, None], vx[:, None],
                                 vy[:, None], vz[:, None], ax[:, None],
                                 ay[:, None], az[:, None],
                                 torch.zeros_like(x[:, None])), 1)
        object_data = object_data.squeeze()
        object_data = object_data.to(device=device)

        all_data.append(object_data)

    # convert_data

    x_list = []
    y_list = []

    for it in trange(5, n_frames - 5):
        for n in range(25):
            x = all_data[n][it, 1]
            y = all_data[n][it, 2]
            z = all_data[n][it, 3]
            vx = all_data[n][it, 4]
            vy = all_data[n][it, 5]
            vz = all_data[n][it, 6]

            tmp = torch.stack(
                [torch.tensor(n,device=device), x, y, z, vx, vy, vz, torch.tensor(n,device=device), torch.tensor(0,device=device), torch.tensor(0,device=device), torch.tensor(0,device=device)])
            if n == 0:
                object_data = tmp[None, :]
            else:
                object_data = torch.cat((object_data, tmp[None, :]), 0)

            ax = all_data[n][it+1, 7]
            ay = all_data[n][it+1, 8]
            az = all_data[n][it+1, 9]
            tmp = torch.stack([ax, ay, az])
            if n == 0:
                acc_data = tmp[None, :]
            else:
                acc_data = torch.cat((acc_data, tmp[None, :]), 0)

        x_list.append(object_data.to(device))
        y_list.append(acc_data.to(device))

    for run in range(2):
        torch.save(x_list, f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_gravity_solar_system/x_list_{run}.pt')
        torch.save(y_list, f'/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/graphs_gravity_solar_system/y_list_{run}.pt')


def load_shrofflab_celegans(
        file_path,
        *,
        replace_missing_cpm=None,
        device='cuda:0'
):
    """
    Load the Shrofflab C. elegans data from a CSV file and convert it to a PyTorch tensor.

    :param file_path: The path to the CSV file.
    :param replace_missing_cpm: If not None, replace missing cpm values (NaN) with this value.
    :param device: The PyTorch device to allocate the tensors on.
    :return: A tuple consisting of:
     * A :py:class:`TimeSeries` object containing the loaded data for each time point.
     * The names of the cells in the data.
    :raises ValueError: If the time series are not part of the same timeframe or if too many cells have abnormal time
    series lengths.
    """

    # Load the data from the CSV file and clean it a bit:
    # - drop rows with missing time values (occurs only at the end of the data)
    # - fill missing cpm values (don't interpolate because data is missing at the beginning or end)
    column_descriptors = {
        "x": CsvDescriptor(filename=file_path, column_name="x", type=np.float32, unit=u.micrometer),
        "y": CsvDescriptor(filename=file_path, column_name="y", type=np.float32, unit=u.micrometer),
        "z": CsvDescriptor(filename=file_path, column_name="z", type=np.float32, unit=u.micrometer),
        "t": CsvDescriptor(filename=file_path, column_name="time", type=np.float32, unit=u.day),
        "cell": CsvDescriptor(filename=file_path, column_name="cell", type=str, unit=u.dimensionless_unscaled),
        "cpm": CsvDescriptor(filename=file_path, column_name="log10 mean cpm", type=np.float32, unit=u.dimensionless_unscaled),
    }
    raw_data = load_csv_from_descriptors(column_descriptors)
    print(f"Loaded {raw_data.shape[0]} rows of data, dropping rows with missing time values...")
    raw_data.dropna(subset=["t"], inplace=True)
    print(f"Remaining: {raw_data.shape[0]} rows")
    if replace_missing_cpm is not None:
        print(f"Filling missing cpm values with {replace_missing_cpm}...")
        raw_data.fillna(replace_missing_cpm, inplace=True)

    # Find the indices where the data for each cell begins (time resets)
    time_reset = np.where(np.diff(raw_data["t"]) < 0)[0] + 1
    timeseries_boundaries = np.hstack([0, time_reset, raw_data.shape[0]])
    n_timepoints = np.diff(timeseries_boundaries).astype(int)
    n_normal_timepoints = np.median(n_timepoints).astype(int)
    start_time, end_time = np.min(raw_data["t"]), np.max(raw_data["t"]) + 1
    n_cells = len(n_timepoints)

    # Sanity checks to make sure the data is not too bad
    n_normal_data = np.count_nonzero(n_timepoints == n_normal_timepoints)
    cell_names = raw_data["cell"].values[timeseries_boundaries[:-1]]
    if (end_time - start_time) != n_normal_timepoints:
        raise ValueError("The time series are not part of the same timeframe.")
    if n_normal_data < 0.5 * n_cells:
        raise ValueError("Too many cells have abnormal time series lengths.")
    if n_normal_data != n_cells:
        abnormal_data = n_timepoints != n_normal_timepoints
        abnormal_cells = cell_names[abnormal_data]
        print(f"Warning: incomplete time series data for {abnormal_cells}")

    # Put values into a TimeSeries object
    relevant_fields = ["x", "y", "z", "cpm", "cell_id"]
    tensors_np = {name: np.nan * np.ones((n_cells * n_normal_timepoints)) for name in relevant_fields}
    time_idx = (raw_data["t"].to_numpy() - start_time).astype(int)
    cell_id = np.repeat(np.arange(n_cells), n_timepoints)
    raw_data.insert(0, "cell_id", cell_id)
    idx = np.ravel_multi_index((cell_id, time_idx), (n_cells, n_normal_timepoints))
    tensors = {}
    for name in relevant_fields:
        tensors_np[name][idx] = raw_data[name].to_numpy()
        split_tensors = np.squeeze(
            np.hsplit(tensors_np[name].reshape((n_cells, n_normal_timepoints)), n_normal_timepoints))
        tensors[name] = [torch.tensor(t, device=device) for t in split_tensors]

    time = torch.arange(start_time, end_time)
    data = [Data(
        time=time[i],
        cell_id=tensors["cell_id"][i],
        pos=torch.stack([tensors["x"][i], tensors["y"][i], tensors["z"][i]], dim=1),
        cpm=tensors["cpm"][i],
    ) for i in range(n_normal_timepoints)]
    time_series = TimeSeries(time, data)

    # Compute the velocity and the derivative of the gene expressions and add them to the time series
    velocity = time_series.compute_derivative('pos')
    d_cpm = time_series.compute_derivative('cpm')
    for i, data in enumerate(time_series):
        data.velocity = velocity[i]
        data.d_cpm = d_cpm[i]

    return time_series, cell_names


def load_celegans_gene_data(
        file_path,
        *,
        coordinate_system: Literal["cartesian", "polar"] = "cartesian",
        device='cuda:0'
):
    """
    Load C. elegans cell data from an HDF5 file (positions and gene expressions) and convert it to a PyTorch tensor.

    :param file_path: The path to the HDF5 file.
    :param coordinate_system: The coordinate system to use for the positions (either "cartesian" or "polar").
    :param device: The PyTorch device to allocate the tensors on.
    :return: A tuple consisting of:
     * A :py:class:`TimeSeries` object containing the loaded data for each time point.
     * A :py:class:`pandas.DataFrame` object containing information about the cells.
    """

    # Load cell information from the HDF5 file (metadata string) into pandas dataframe
    print(f"Loading data from '{file_path}'...")
    file = h5py.File(file_path, 'r')
    cell_info_raw = file["cellinf"][0][0].decode("utf-8")
    cell_info_raw = cell_info_raw.replace("false", "False").replace("true", "True")
    cell_info_raw = eval(cell_info_raw)

    names = [info.pop('name') for info in cell_info_raw]
    cell_info = pd.DataFrame(cell_info_raw, index=names)

    # There are two time series: one for the gene expressions (sparse) and one for the positions (dense)
    # Compute intersection of both time series and interpolate gene expressions where they are not defined
    gene_time = file["gene_time"][0]
    pos_time = file["pos_time"][0]
    min_t = max(gene_time[0], pos_time[0])
    max_t = min(gene_time[-1], pos_time[-1])
    time = np.arange(min_t, max_t + 1)
    pos_overlap = np.isin(pos_time, time)
    genes_overlap = np.isin(gene_time, time)

    # Assign positions
    match coordinate_system:
        case "cartesian":
            positions = file["pos_xyz"][pos_overlap]
        case "polar":
            positions = file["pos_rpz"][pos_overlap]
        case _:
            raise ValueError(f"Invalid coordinate system '{coordinate_system}'")

    # Interpolate gene expressions by piecewise linear spline
    gene_data = file["gene_CPM"][genes_overlap]
    t = gene_time[genes_overlap]
    f = make_interp_spline(t, gene_data, k=1, axis=0, check_finite=False)

    # Due to NaNs in the gene data, the interpolation is not perfect; make sure at least original data is present
    genes_are_present = np.isin(time, gene_time)
    interpolated_to_present_data = -np.ones_like(time, dtype=int)
    interpolated_to_present_data[genes_are_present] = np.arange(np.count_nonzero(genes_overlap))

    # Bundle everything in a TimeSeries object
    data = []
    for t in trange(len(time)):
        if genes_are_present[t]:
            interpolated_gene_data = gene_data[interpolated_to_present_data[t]]
        else:
            interpolated_gene_data = f(time[t])
        data.append(Data(
            time=time[t],
            pos=torch.tensor(positions[t], device=device),
            gene_cpm=torch.tensor(interpolated_gene_data.T, device=device),
        ))
    time_series = TimeSeries(torch.tensor(time, device=device), data)
    file.close()

    # Compute the velocity and the derivative of the gene expressions and add them to the time series
    velocity = time_series.compute_derivative('pos')
    d_cpm = time_series.compute_derivative('gene_cpm')
    for i, data in enumerate(time_series):
        data.velocity = velocity[i]
        data.d_gene_cpm = d_cpm[i]

    return time_series, cell_info


def load_agent_data(
        data_directory,
        *,
        device='cuda:0'
):
    """
    Load simulated agent data and convert it to a time series.

    :param data_directory: The directory containing the agent data.
    :param device: The PyTorch device to allocate the tensors on.
    :return: A tuple consisting of:
     * A :py:class:`TimeSeries` object containing the loaded data for each time point.
     * A 2D grid of the signal that the agents are responding to.
    """

    # Check how many files (each a timestep) there are
    print(f"Loading data from '{data_directory}'...")
    files = os.listdir(data_directory)
    file_name_pattern = re.compile(r'particles\d+.txt')
    n_time_points = sum(1 for f in files if file_name_pattern.match(f))

    # Load the data from text (csv) files and convert everything to to Data objects (all fields are float32)
    dtype = {
        "x": np.float32,
        "y": np.float32,
        "internal": np.float32,
        "orientation": np.float32,
        "reversal_timer": np.int64,
        "state": np.int64
    }

    data = []
    time = torch.arange(1, n_time_points + 1, device=device)
    for i in trange(n_time_points):
        file_path = os.path.join(data_directory, f"particles{i + 1}.txt")
        time_point = pd.read_csv(file_path, sep=",", names=list(dtype.keys()), dtype=dtype)
        position = torch.stack([torch.tensor(time_point["x"].to_numpy(), device=device),
                                torch.tensor(time_point["y"].to_numpy(), device=device)], dim=1)
        data.append(Data(
            time=time[i],
            pos=position,
            internal=torch.tensor(time_point["internal"].to_numpy(), device=device),
            orientation=torch.tensor(time_point["orientation"].to_numpy(), device=device),
            reversal_timer=torch.tensor(time_point["reversal_timer"].to_numpy(), dtype=torch.float32, device=device),
            state=torch.tensor(time_point["state"].to_numpy(), dtype=torch.float32, device=device),
        ))

    # Compute the velocity as the derivative of the position and add it to the time series
    time_series = TimeSeries(time, data)
    velocity = time_series.compute_derivative('pos')
    for i, data in enumerate(time_series):
        data.velocity = velocity[i]

    # Load the signal
    signal = np.loadtxt(os.path.join(data_directory, "signal.txt"))
    signal = torch.tensor(signal, device=device)

    return time_series, signal


def ensure_local_path_exists(path):
    """
    Ensure that the local path exists. If it doesn't, create the directory structure.

    :param path: The path to be checked and created if necessary.
    :return: The absolute path of the created directory.
    """

    os.makedirs(path, exist_ok=True)
    return os.path.join(os.getcwd(), path)


@dataclass
class CsvDescriptor:
    """A class to describe the location of data in a dataset as a column of a CSV file."""
    filename: str
    column_name: str
    type: np.dtype
    unit: Unit


def load_csv_from_descriptors(
        column_descriptors: Dict[str, CsvDescriptor],
        **kwargs
) -> pd.DataFrame:
    """
    Load data from a CSV file based on a set of column descriptors.

    :param column_descriptors: A dictionary mapping field names to CsvDescriptors.
    :param kwargs: Additional keyword arguments to pass to pd.read_csv.
    :return: A pandas DataFrame containing the loaded data.
    """
    different_files = set(descriptor.filename for descriptor in column_descriptors.values())
    columns = []

    for file in different_files:
        dtypes = {descriptor.column_name: descriptor.type for descriptor in column_descriptors.values()
                  if descriptor.filename == file}
        print(f"Loading data from '{file}':")
        for column_name, dtype in dtypes.items():
            print(f"  - column {column_name} as {dtype}")
        columns.append(pd.read_csv(file, dtype=dtypes, usecols=list(dtypes.keys()), **kwargs))

    data = pd.concat(columns, axis='columns')
    data.rename(columns={descriptor.column_name: name for name, descriptor in column_descriptors.items()}, inplace=True)

    return data


def load_wanglab_salivary_gland(
        file_path: str,
        *,
        device: str = 'cuda:0'
) -> Tuple[TimeSeries, torch.Tensor]:
    """
    Load the Wanglab salivary gland data from a CSV file and convert it to a pytorch_geometric Data object.

    :param file_path: The path to the CSV file.
    :param device: The PyTorch device to allocate the tensors on.
    :return: A :py:class:`TimeSeries` object containing the loaded data for each time point.
    """

    # Load the data of interest from the CSV file
    column_descriptors = {
        'x': CsvDescriptor(filename=file_path, column_name="Position X", type=np.float32, unit=u.micrometer),
        'y': CsvDescriptor(filename=file_path, column_name="Position Y", type=np.float32, unit=u.micrometer),
        'z': CsvDescriptor(filename=file_path, column_name="Position Z", type=np.float32, unit=u.micrometer),
        't': CsvDescriptor(filename=file_path, column_name="Time", type=np.float32, unit=u.day),
        'track_id': CsvDescriptor(filename=file_path, column_name="TrackID", type=np.int64,
                                  unit=u.dimensionless_unscaled),
    }
    raw_data = load_csv_from_descriptors(column_descriptors, skiprows=3)
    raw_tensors = {name: torch.tensor(raw_data[name].to_numpy(), device=device) for name in column_descriptors.keys()}

    # Split into individual data objects for each time point
    t = raw_tensors['t']
    time_jumps = torch.where(torch.diff(t).ne(0))[0] + 1
    time = torch.unique_consecutive(t)
    x = torch.tensor_split(raw_tensors['x'], time_jumps.tolist())
    y = torch.tensor_split(raw_tensors['y'], time_jumps.tolist())
    z = torch.tensor_split(raw_tensors['z'], time_jumps.tolist())
    global_ids, id_indices = torch.unique(raw_tensors['track_id'], return_inverse=True)
    id = torch.tensor_split(id_indices, time_jumps.tolist())

    # Combine the data into a TimeSeries object
    n_time_steps = len(time)
    data = []
    for i in range(n_time_steps):
        data.append(Data(
            time=time[i],
            pos=torch.stack([x[i], y[i], z[i]], dim=1),
            track_id=id[i],
        ))

    time_series = TimeSeries(time, data)

    # Compute the velocity as the derivative of the position and add it to the time series
    velocity, _ = time_series.compute_derivative('pos', id_name='track_id')
    for i in range(n_time_steps):
        data[i].velocity = velocity[i]

    return time_series, global_ids
