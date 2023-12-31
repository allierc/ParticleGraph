"""
A collection of functions for loading data from various sources.
"""

import os
import numpy as np
import torch
import pandas as pd


def load_shrofflab_celegans(
        file_path,
        *,
        replace_missing_cpm=None,
        device='cuda:0'
):
    """
    Load the Shrofflab C. elegans data from a CSV file and convert it to a PyTorch tensor.

    Args:
        file_path (str): The path to the CSV file.
        replace_missing_cpm (float): If not None, replace missing cpm values with this value.
        device (str): The PyTorch device to use for the tensor.

    Returns:
        tensor_list (List[torch.Tensor]): A list of PyTorch tensors containing the loaded data for each time point.
        time (np.ndarray): The time points corresponding to the data.
        cell_names (np.ndarray): The names of the cells in the data.

    Raises: ValueError: If the time series are not part of the same timeframe or if too many cells have abnormal time
    series lengths.
    """

    # Load the data from the CSV file and clean it a bit:
    # - drop rows with missing time values (occurs only at the end of the data)
    # - fill missing cpm values (don't interpolate because data is missing at the beginning or end)
    print(f"Loading data from {file_path}...")
    dtypes = {"time": np.float32, "x": np.float32, "y": np.float32,
              "z": np.float32, "cell": str, "log10 mean cpm": str}
    data = pd.read_csv(file_path, dtype=dtypes)
    print(f"Loaded {data.shape[0]} rows of data, dropping rows with missing time values...")
    data.dropna(subset=["time"], inplace=True)
    print(f"Remaining: {data.shape[0]} rows")
    if replace_missing_cpm is not None:
        print(f"Filling missing cpm values with {replace_missing_cpm}...")
        data.fillna(replace_missing_cpm, inplace=True)

    # Find the indices where the data for each cell begins (time resets)
    time_reset = np.where(np.diff(data["time"]) < 0)[0] + 1
    timeseries_boundaries = np.hstack([0, time_reset, data.shape[0]])
    n_timepoints = np.diff(timeseries_boundaries).astype(int)
    n_normal_timepoints = np.median(n_timepoints).astype(int)
    start_time, end_time = np.min(data["time"]), np.max(data["time"]) + 1
    n_cells = len(n_timepoints)

    # Sanity checks to make sure the data is not too bad
    n_normal_data = np.count_nonzero(n_timepoints == n_normal_timepoints)
    cell_names = data["cell"].values[timeseries_boundaries[:-1]]
    if (end_time - start_time) != n_normal_timepoints:
        raise ValueError("The time series are not part of the same timeframe.")
    if n_normal_data < 0.5 * n_cells:
        raise ValueError("Too many cells have abnormal time series lengths.")
    if n_normal_data != n_cells:
        abnormal_data = n_timepoints != n_normal_timepoints
        abnormal_cells = cell_names[abnormal_data]
        print(f"Warning: incomplete time series data for {abnormal_cells}")

    # Put values into a 3D tensor
    relevant_fields = ["x", "y", "z", "log10 mean cpm"]
    shape = (n_cells, n_normal_timepoints, len(relevant_fields))
    tensor = np.nan * np.ones((shape[0] * shape[1], shape[2]))
    time_idx = (data["time"].values - start_time).astype(int)
    cell_idx = np.repeat(np.arange(n_cells), n_timepoints)
    idx = np.ravel_multi_index((cell_idx, time_idx), (n_cells, n_normal_timepoints))
    for i, name in enumerate(relevant_fields):
        tensor[idx, i] = data[name].values
    tensor = np.transpose(tensor.reshape(shape), (1, 0, 2))

    # Compute the time derivatives and concatenate such that columns correspond to:
    # x, y, z, d/dt x, d/dt y, d/dt z, cpm, d/dt cpm
    tensor_gradient = tensor - np.roll(tensor, 1, 0)
    tensor_gradient[0, :, :] = np.nan
    tensor = np.concatenate([tensor[:, :, 0:3], tensor_gradient[:, :, 0:3],
                             tensor[:, :, 3:4], tensor_gradient[:, :, 3:4]], axis=2)

    # Put all the time points into a separate tensor
    tensor_list = []
    for i in range(n_normal_timepoints):
        cell_tensor = tensor[i]
        cell_ids = np.where(~np.isnan(cell_tensor[:, 0]))[0]
        cell_tensor = np.column_stack((cell_ids, cell_tensor[cell_ids, :]))
        tensor_list.append(torch.tensor(cell_tensor, device=device))

    time = np.arange(start_time, end_time)

    return tensor_list, time, cell_names


def ensure_local_path_exists(path):
    """
    Ensure that the local path exists. If it doesn't, create the directory structure.

    Args:
        path (str): The path to be checked and created if necessary.

    Returns:
        str: The absolute path of the created directory.
    """

    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(os.getcwd(), path)
