import numpy as np
import argparse
from sklearn import metrics
from utils import *
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Link prediction.")
    parser.add_argument('--input', nargs='?',
                        help='Input embedding path.')
    return parser.parse_args()


def link_predict(filename, sim, dataset):
    f = open(filename, 'r', encoding="utf8")

    embs = {}
    f.readline()

    while True:
        line = f.readline().strip().split()
        if line == []:
            break

        name = line[0]
        del line[0]

        vect = []
        for m in line:
            try:
                vect.append(float(m))
            except BaseException:
                vect.append(0)
        embs[name] = vect

    test = []
    test_neg = []

    f1 = open('hypergraph/' + dataset + '/test.edgelist', 'r', encoding="utf8")

    while True:
        line = f1.readline().strip().split()
        if line == []:
            break

        name = line[0]
        del line[0]

        vect = []
        for m in line:
            vect.append(m)
        test.append(vect)

    f1 = open('hypergraph/' + dataset + '/test_negative.edgelist', 'r', encoding="utf8")

    while True:
        line = f1.readline().strip().split()
        if line == []:
            break

        vect = []
        for m in line:
            vect.append(m)
        test_neg.append(vect)

    y = []
    pred = []

    for edge in test:
        flag = False
        for node in edge:
            if embs.get(node) == None:
                flag = True
                break
        if flag:
            continue
        d = hyperedge_dist(edge, embs, sim)
        y.append(1)
        pred.append(d)

    for edge in test_neg:
        flag = False
        for node in edge:
            if embs.get(node) == None:
                flag = True
                break
        if flag:
            continue
        d = hyperedge_dist(edge, embs, sim)
        y.append(0)
        pred.append(d)

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    if sim == COS:
        return metrics.auc(fpr, tpr), fpr, tpr
    else:
        return 1 - metrics.auc(fpr, tpr), fpr, tpr


def main(args):
    # funcs = [L1D, L2D, L3D, L4D, L5D, L6D, L7D, COS]
    funcs = [L1D, L2D, COS]
    dataset = args.input.split('/')[1]  # extract dataset name
    print(dataset)
    for sim in funcs:
        print(sim.__name__, end=': ')
        auc, tpr, fpr = link_predict(args.input, sim, dataset)
        print(auc)
        # plt.plot(fpr, tpr, color = 'b', label='ROC (area = {0:.2f})'.format(auc), lw=2)
 
        # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        # plt.title('ROC Curve')
        # plt.legend(loc="lower right")
        # plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)



