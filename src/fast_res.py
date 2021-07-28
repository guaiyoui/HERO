from math import *
import numpy as np
import matplotlib.pyplot as plt
from hypergraph import *
from tqdm import tqdm
import time

from numba import njit

import sys

np.set_printoptions(threshold=np.inf)


def _initialization(data, W, rows, columns, k_dim, a, b, c):
    """Initializes latent factor matrixes.
    """
    p = np.random.normal(0, 0.1, (rows, k_dim))
    # q = np.random.normal(0, 0.1, (k_dim, columns))
    alpha = 2**np.random.randint(0, 4, size=(k_dim, columns))
    q = np.empty(alpha.shape)
    for k in range(k_dim):
        q[k] = np.random.dirichlet(alpha[k])

    value = np.zeros((columns, 3))
    # print(a,b,c)
    
    value[:, 0] = a
    value[:, 1] = b
    value[:, 2] = c

    return value, p, q

@njit
def _run_epoch(R, value, W, P, Q, k_dim, alpha, beta):
    # print("Running")
    rows, columns = R.shape
    for j in range(columns):
        for i in range(rows):
            row_index = R[i][j]
            col_index = i
            eij = value[i][j] - P[row_index, :]@Q[:, col_index]
            for k in range(k_dim):
                P[row_index][k] = P[row_index][k] + alpha * (2 * eij * Q[k][col_index] - beta * P[row_index][k])
                Q[k][col_index] = Q[k][col_index] + alpha * (2 * eij * P[row_index][k] - beta * Q[k][col_index])
    return P, Q

def Matrix_Factorization(dataset, data, W, row_num, col_num, dimensions=32, steps=10000, alpha=0.00025, beta=0.055, a = 1, b = 1, c = 1):
    
    value, P, Q = _initialization(data, W, row_num, col_num, dimensions, a, b, c)

    start = time.time()
    for epoch in tqdm(range(steps)):
        P, Q = _run_epoch(data, value, W, P, Q, dimensions, alpha, beta)

    end = time.time()
    print(P.shape, Q.shape)
    return start, end, P, Q

def Update_MF(P, Q, G, update_data, dimensions=32, a = 1, b = 1, c = 1):
    rows, cols = update_data.shape
    _type = 0
    node_id = -1
    
    weights = np.array([a, b, c])
    start = time.time()
    for i in range(rows):
        if update_data[i][0] not in G.nodes():
            _type = 1 #type B update
            node_id = update_data[i][0]
        elif update_data[i][1] not in G.nodes():
            _type = 1 #type B update
            node_id = update_data[i][1]
        elif update_data[i][2] not in G.nodes():
            _type = 1 #type B update
            node_id = update_data[i][2]
        else:
            _type = 2 #type A update

        if _type == 1:
            # print("here")
            if node_id > max(G.nodes()):
                p_add = np.random.normal(0, 0.01, (node_id - max(G.nodes()), dimensions))
                P = np.row_stack((P,p_add))
                # print("here")
            edge_name = str(len(list(G.edges()))+1)
            G.add_edge(edge_name, update_data[i])
            q_add = np.random.normal(0, 0.01, (1, dimensions))
            for j in range(3):
                q_add[0][j] = weights[j]/P[update_data[i][j]][j]
            Q = np.column_stack((Q, q_add[0]))
            P[node_id][0] = weights[2]/q_add[0][1]
            # edge_name = str(len(list(G.edges()))+1)
            # G.add_edge(edge_name, update_data[i])


            # print("here")
            # print(node_id)
        if _type == 2:
            edge_name = str(len(list(G.edges()))+1)
            G.add_edge(edge_name, update_data[i])
            q_add = np.random.normal(0, 0.01, (1, dimensions))
            # print('--------------------------------')
            # print(q_add[0])
            for j in range(3):
                q_add[0][j] = weights[j]/P[update_data[i][j]][j]
            # print(q_add[0])
            # print(q_add)
            # Q.append(q_add)
            Q = np.column_stack((Q, q_add[0]))
    end = time.time()
    return start, end, P, Q


def stror_inf(dataset, start, end, start_u, end_u, P, Q):

    result = open('../emb/'+dataset+'/result_123_speedtest_for_normal.txt', 'w')
    emb = open('../emb/'+dataset+'/SPREE.emb', 'w')

    result.write("embedding time:"+str(end-start)+'\n')
    result.write("updating time:"+str(end_u-start_u)+'\n')
    # result.write("-------------P matrix-------------------\n")
    # result.write(str(np.shape(P))+'\n')
    # result.write(str(P)+'\n')
    emb.write(str(P) + '\n')
    # result.write("-------------Q matrix-------------------\n")
    # result.write(str(np.shape(Q))+'\n')
    # result.write(str(Q)+'\n')
    
    return


def read_graph(dataset,Mtype):
    if Mtype == 0:
        f = open('../hypergraph/'+dataset+'/train.edgelist', 'r', encoding='utf8')
    else:
        f = open('../hypergraph/'+dataset+'/train_.edgelist', 'r', encoding='utf8')
    graph_type = f.readline().strip()
    nums_type = None
    if graph_type == '1':  # heterogeneous
        nums_type = f.readline().split()
        nums_type = list(map(int, nums_type))
    G = Hypergraph(graph_type, nums_type)
    for line in tqdm(f.readlines()):
        line = line.split()
        edge_name = line[0]
        G.add_edge(edge_name, map(int, line[1:]))
    f.close()
    R, W = Graph2List(G)
    R = np.array(R)
    row_num, col_num = getSizes(G)

    # print(G.edges())

    # new_edge = (111,167,220)
    # edge_name = str(len(list(G.edges()))+1)
    # print(edge_name)
    # G.add_edge(edge_name, new_edge)
    # edge_name = str(len(list(G.edges()))+1)
    # print(G.edges())
    # if new_edge in G.edges():
    #     print("yes")
    # new_node = 7378
    # # print(G.nodes())
    # if new_node in G.nodes():
    #     print("yes")
    return R, W, row_num, col_num, G

def read_update(dataset):
    f = open('../hypergraph/'+dataset+'/add_.edgelist', 'r', encoding='utf8')
    graph_type = f.readline().strip()
    nums_type = None
    if graph_type == '1':  # heterogeneous
        nums_type = f.readline().split()
        nums_type = list(map(int, nums_type))
    G = Hypergraph(graph_type, nums_type)
    for line in tqdm(f.readlines()):
        line = line.split()
        edge_name = line[0]
        G.add_edge(edge_name, map(int, line[1:]))
    f.close()
    R, W = Graph2List(G)
    R = np.array(R)
    # print(np.shape(R))
    return R