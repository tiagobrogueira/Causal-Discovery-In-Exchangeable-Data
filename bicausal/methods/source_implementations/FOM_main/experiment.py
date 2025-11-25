# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import minmax_scale
from models import build_model
from data.utils import sampling, cal_decision_rate
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn


def run_SYN(data_path, algorithm='FOM'):
    print('='*25)
    print(f'= {algorithm}')
    print('='*25)
    files = [
        os.path.join(data_path, file_name)
        for file_name in os.listdir(data_path) if file_name[-4:] == '.csv'
    ]
    assert len(files) != 0
    scores = []
    for i, f in enumerate(files):
        print('Task %d: ' % (i + 1))
        print('ground true: x-->y')
        df = pd.read_csv(f)
        x = df['x']
        y = df['y']
        x, y = sampling(x, y, 1000)  # reduce sample size for hgpr
        x, y = minmax_scale(x, (-1.0, 1.0)), minmax_scale(y, (-1.0, 1.0))
        x = np.reshape(x, [-1, 1])
        y = np.reshape(y, [-1, 1])
        model = build_model(algorithm)
        score = model.predict(x, y)
        scores.append(score)
        if score > 0.0:
            print('True')
        else:
            print('False')
        print('-' * 25)
    print(f'total num of causal pair: {len(files)}')
    acc = 100. * np.count_nonzero(np.array(scores) > 0.0) / float(len(files))
    print('accaury: %.3f' % acc)
    return acc, np.array(scores)


def run_CEP(data_path, algorithm='FOM'):
    print('='*25)
    print(f'=  {algorithm}')
    print('='*25)
    files = [
        os.path.join(data_path, 'pair%s.txt' % (str(i).zfill(4)))
        for i in range(1, 101)
    ]
    dfMeta = pd.read_csv(os.path.join(data_path, 'pairmeta.txt'),
                         header=None,
                         delim_whitespace=True)

    weight = dfMeta[5].values
    dire = dfMeta[4].values
    w = 0
    ws = 0
    fails = []
    scores = []
    labels = []
    weights = []
    delete = [43, 44, 45, 46, 65, 66, 67, 76, 84] # delete causal pair without heteroskedasticity
    cnt = 0
    for i, f in enumerate(files):
        if weight[i] != 0 and (i + 1) not in delete:
            print('Task %d' % (i + 1))
            if dire[i] == 2:
                print('ground true: x-->y')
                labels.append(1)
            else:
                print('ground true: y-->x')
                labels.append(0)
            ws += weight[i]
            cnt += 1
            weights.append(weight[i])
            df = pd.read_csv(f, header=None, delim_whitespace=True)
            x = df[0].values
            y = df[1].values
            x, y = sampling(x, y, 1000)
            x, y = minmax_scale(x), minmax_scale(y)
            x = np.reshape(x, [-1, 1])
            y = np.reshape(y, [-1, 1])
            model = build_model(algorithm)
            score = model.predict(x, y)
            scores.append(score)
            if (score > 0. and dire[i] == 2) or (score < 0. and dire[i] == 1):
                print('True')
                w += weight[i]
            else:
                print('False')
                fails.append(i + 1)
            print('-' * 25)
    print(f'total num of causal pair: {cnt}')
    acc = w / ws
    print('accaury: %.3f' % acc)
    return acc, fails, (scores, labels, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment for causal discovery.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--experiment_type', type=str, default='CEP',
                        help='expriment type, CEP or SYN')
    parser.add_argument('--dataset', type=str, default='./datasets/CEP/',
                        help='path of dataset')
    parser.add_argument('--algorithm', type=str, default='FOM',
                        help='algorithm name. one of ANM, IGCI, RECI, FOM')
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.experiment_type == 'CEP':
        # CEP dataset
        acc, fails, result = run_CEP(args.dataset, args.algorithm)
        # steps, decision_rate = cal_decision_rate(result[0], result[1], result[2])
        # print(decision_rate)
    if args.experiment_type == 'SYN':
        # synithetic dataset
        acc, scores = run_SYN(args.dataset, args.algorithm)
        # steps, decision_rate = cal_decision_rate(scores)
        # print(decision_rate)

