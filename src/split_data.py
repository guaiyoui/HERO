import numpy as np
from hypergraph import *
from tqdm import tqdm

def split_data_1(dataset, ratio):

    data_origin = open('../hypergraph/'+dataset+'/train.edgelist','r', encoding='utf8')
    data_train = open('../hypergraph/'+dataset+'/train_.edgelist','w', encoding='utf8')
    data_add = open('../hypergraph/'+dataset+'/add_.edgelist','w', encoding='utf8')

    graph_type = data_origin.readline().strip()
    nums_type = data_origin.readline().split()
    nums_type = list(map(int, nums_type))

    # max_index = nums_type.index(max(nums_type))


    G = Hypergraph(graph_type, nums_type)
    lines = data_origin.readlines()
    
    addEdge_num = int(ratio*len(lines))
    print(addEdge_num)
   
    max_index = len(nums_type)
    high_index = sum(nums_type[0:max_index])-1
    num_add = 0
    p1 = []
    p2 = []

    for line in tqdm(lines):
        line = line.split()
        edge_name = line[0]
        G.add_edge(edge_name, map(int, line[1:]))

    cnt = 0
    added_index = []
    while(1):
        index_add = high_index
        line_index = -1
        # print(high_index)
        for line in lines:
            line_index += 1
            line = line.split()
            edge_name = line[0]
            edge = list(map(int, line[1:]))
            if edge[max_index-1] == index_add:
                added_index.append(line_index)
                p1.append(edge)
                cnt += 1
            # else:
            #     p2.append(edge)
            if cnt == addEdge_num:
                break
        # print(cnt)
        if cnt == addEdge_num:
            break
        
        high_index -= 1
        num_add += 1

    line_index = -1
    for line in tqdm(lines):
            line_index += 1
            line = line.split()
            edge_name = line[0]
            edge = list(map(int, line[1:]))
            if line_index not in added_index:
                p2.append(edge)

    data_train.write("1\n")
    
    nums_type[max_index-1] -= num_add
    data_train.write(str(nums_type[0]))
    for i in range(1, len(nums_type)):
        data_train.write(" "+str(nums_type[i]))
    data_train.write("\n")
    
    index = 0
    for i in p2:
        data_train.write(str(index)+' ')
        data_train.write(str(i[0]))
        for j in range(1, len(i)):
            data_train.write(" "+str(i[j]))
        data_train.write("\n")
        index+=1
    
    data_add.write("1\n")
    nums_type[max_index-1] = num_add
    data_add.write(str(nums_type[0]))
    for i in range(1, len(nums_type)):
        data_add.write(" "+str(nums_type[i]))
    data_add.write("\n")
    
    index = 0
    for i in p1:
        data_add.write(str(index)+' ')
        data_add.write(str(i[0]))
        for j in range(1, len(i)):
            data_add.write(" "+str(i[j]))
        data_add.write("\n")
        index+=1


# split_data_1('GPS', 0.15)
# split_data_1('MovieLens', 0.15)
# split_data_1('Drugs', 0.15)
# split_data_1('Wordnet', 0.15) 

# split_data_1('GPS', 0.20)
# split_data_1('MovieLens', 0.20)
# split_data_1('Drugs', 0.20)
split_data_1('Wordnet', 0.20)  