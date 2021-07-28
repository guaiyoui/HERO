import numpy as np
import math

# transform data into line format so that it can be used by downstream tasks easily.

def trans_data(dataset):
    np.set_printoptions(threshold=np.inf)
    
    f = open('../emb/'+dataset+'/SPREE.emb','r', encoding='utf8')
    result = open('../emb/'+dataset+'/Res.emb','w', encoding='utf8')

    dict = {'GPS': 221, 'Drugs': 7487, 'MovieLens': 15593, 'Wordnet':81520}
    emb = np.zeros((dict[dataset], 32))

    index = 0
    cnt = 0

    for line in f.readlines():
        line = line.replace('[',' ')
        line = line.replace(']',' ')
        line = line.replace('\n',' ')
        line = line.split(' ')
        for item in line:
            if item != '':
                item = float(item)
                if math.isnan(item):
                    item = 0.0
                emb[index][cnt] = item
                cnt += 1
        if cnt == 32:
            cnt = 0
            index += 1
    
    result.write(str(dict[dataset])+" 32\n")
    for i in range(0,dict[dataset]):
        result.write(str(i))
        for j in range(0,32):
            result.write(' '+str(emb[i][j]))
        result.write('\n')

    return