'''
Reference implementation of 

Author: Jianwei Wang

For more details, refer to the paper:
Representation Learning for Hyper-Networks
JianweiWang
'''


import argparse
import time
import numpy as np
from hypergraph import *
from fast_res import *
from transform_data import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run algorithm.")

    parser.add_argument('--dataset', nargs='?',
                        help='dataset name.')

    parser.add_argument('--dimensions', type=int, default=32,
                        help='Number of dimensions. Default is 32.')

    parser.add_argument('--iter', default=15000, type=int,
                        help='Number of iteration. Default is 40000.')
    
    parser.add_argument('--alpha', type=float, default=0.0002,
                        help='L2 normalization parameter. Default is 0.0002.')

    parser.add_argument('--beta', type=float, default=0.02,
                        help='gradient descent parameter. Default is 0.02.')
                    
    parser.add_argument('--a', type=float, default=1,
                        help='the parameter of first column. Default is 1.')
    parser.add_argument('--b', type=float, default=2,
                        help='the parameter of second column. Default is 2.')
    parser.add_argument('--c', type=float, default=3,
                        help='the parameter of third column. Default is 3.')
    
    parser.add_argument('--Mtype', type=int, default=0,
                        help='embedding or update module. Default is 0 which means embedding.')
    
    return parser.parse_args()


def main(args):
    dataset = args.dataset
    print("="*50)
    dimensions = args.dimensions
    iteration = args.iter
    a = args.a
    b = args.b
    c = args.c
    Mtype = args.Mtype

    data, W, row_num, col_num, G = read_graph(dataset, Mtype)
    start, end, P, Q = Matrix_Factorization(dataset, data, W, row_num, col_num, dimensions, steps = iteration, a = a, b = b, c = c)
    start_u = 0
    end_u = 0
    if Mtype == 1:
        print("here")
        update_data = read_update(dataset)
        start_u, end_u, P, Q = Update_MF(P, Q, G, update_data, dimensions, a, b, c)
    
    print("="*50)
    
    stror_inf(dataset, start, end, start_u, end_u, P, Q)
    trans_data(dataset)

if __name__ == "__main__":
    args = parse_args()
    main(args)


            
