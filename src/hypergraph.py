import numpy as np
from tqdm import tqdm
from itertools import combinations
import copy
import random

class Hypergraph(object):
    def __init__(self,graph_type='0',nums_type=None):
        self._nodes = {}  # node set
        self._edges = {}  # edge set (hash index)
        self.graph_type = graph_type  # graph type, homogeneous:0, heterogeneous:1
        self.nums_type = nums_type  # for heterogeneous graph, number of different node type
        self.cumsum = np.cumsum(self.nums_type) if self.graph_type=='1' else None  # cumsum of nums_type

    def add_edge(self, edge_name, e):
        '''
        Add a hyperedge.
        edge_name: name of hyperedge
        edge: node list of hyperedge
        weight: weight of hyperedge
        '''
        edge = tuple(sorted(e))
        self._edges[edge] = self._edges.get(edge,0)+1
        # print(edge)
        for v in edge:
            node_dict = self._nodes.get(v, {})
            neighbors = node_dict.get('neighbors', set())
            for v0 in edge:
                if v0!=v:
                    neighbors.add(v0)
            node_dict['neighbors'] = neighbors
            if self.graph_type=='1':
                for i,k in enumerate(self.cumsum):
                    if int(v) < k:
                        break
                node_dict['type'] = i

            self._nodes[v] = node_dict

    def edge_weight(self, e):
        '''weight of weight e'''
        return self._edges.get(e,0)

    def nodes(self):
        '''node set'''
        return self._nodes.keys()

    def edges(self):
        '''edge set'''
        return self._edges.keys()

    def neighbors(self, n):
        '''neighbors of node n'''
        return self._nodes[n]['neighbors']

    def node_type(self, n):
        '''type of node n'''
        return self._nodes[n]['type']


def normalization(data):
    _range = np.max(data) - np.min(data)
    # print(_range)
    return (data - np.min(data)) / _range + 0.5



def Graph2List(G):
    edges = list(G.edges())
    edges = np.array(edges)
    W = np.zeros(len(G.nodes()))

    # print(len(G.nodes()))

    # for i in range(len(G.nodes())):
    #     W[i] = len(G.neighbors(i))

    W = len(G.nodes())* W/np.sum(W)
    W = normalization(W)
    # print(W)
    # print(Find_Kth_max(W, int(len(G.nodes()) * 0.10)))
    return edges, W


def getSizes(G):
    edges = list(G.edges())
    # node_num = len(G.nodes())
    # print(G.nodes())
    # print(max(G.nodes()))
    # print(max(G.nodes()))
    return max(G.nodes())+1, len(edges)


