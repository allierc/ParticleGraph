
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
from ParticleGraph.utils import *

class PDE_N9(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-neuron-dependent
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    dv : float
    the update rate of the voltages (dim 1)
        
    """

    def __init__(self, aggr_type='add', p=[], f=[]):
        super(PDE_N9, self).__init__(aggr=aggr_type)

        self.p = p      # dict of parameters alpha_ij, sigma_ij, Nij, dimension NxN   tau_i, V_i_rest dimension N
        self.f = f      # function f()

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index

        cell_type = x[:, 5].long()
        v = x[:, 6:7]
        v_rest = self.p['V_i_rest']
        e = x[:, 7:8]

        msg = self.propagate(edge_index, v=v, cell_type=cell_type)

        tau = self.p['tau_i']
        dv = (-v + msg + e + v_rest) / tau

        return dv


    def message(self, edge_index_i, edge_index_j, v_j, cell_type_i, cell_type_j):

        W = self.p['alpha_ij'] * self.p['sigma_ij'] * self.p['Nij']

        return W[edge_index_i, edge_index_j][:, None] * self.f(v_j)

