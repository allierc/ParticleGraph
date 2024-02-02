from torch_geometric.data import Data


class GraphTimeSeries:
    """
    A class that has a list of torch_geometric.data.Data objects and a feature name map
    """

    _data: list[Data]
    _feature_index: dict[str, int]
    _time_points: list[int]

    def __init__(self, data_list, *, feature_names, time_points=None):
        self._time_points = self._validate(data_list, feature_names, time_points)
        self._data = data_list
        self._feature_index = {feature: i for i, feature in enumerate(feature_names)}

    def _validate(self, data_list, feature_names, time_points):
        n_data = len(data_list)
        if n_data == 0:
            raise ValueError("The data list should not be empty")
        if not n_data == len(feature_names):
            raise ValueError("The number of feature names should match the number of data objects")
        if time_points is not None:
            if not n_data == len(time_points):
                raise ValueError("The number of time points should match the number of data objects")
            t = time_points
        else:
            t = list(range(n_data))
        return t

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get_feature_names(self):
        return list(self._feature_index.keys())

    def get_indices_for(self, feature_name):
        if isinstance(feature_name, str):
            return self._feature_index[feature_name]
        else:
            return [self._feature_index[name] for name in feature_name]

    def get_time_for(self, idx):
        return self._time_points[idx]
