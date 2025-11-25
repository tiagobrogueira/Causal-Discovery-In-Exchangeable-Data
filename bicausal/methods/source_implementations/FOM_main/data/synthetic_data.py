# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline as sp
from sklearn.preprocessing import scale, minmax_scale
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt
import os


def check_dir(outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)


def cause(n, k, p1, p2):
    g = GMM(k, covariance_type='diag')
    g.means_ = p1 * np.random.randn(k, 1)
    g.covariances_ = np.power(abs(p2 * np.random.randn(k, 1) + 1), 2)
    g.weights_ = abs(np.random.rand(k, ))
    g.weights_ = g.weights_ / sum(g.weights_)
    x, _ = g.sample(n)
    return scale(x)


def noise(n, v):
    return v * np.random.rand(1) * np.random.randn(n, 1)


def noise_(n, v):
    return v * np.random.randn(n, 1)


def mechanism(x, d):
    g = np.linspace(min(x) - np.std(x), max(x) + np.std(x), d)
    return sp(g, np.random.randn(d))(x.flatten())[:, np.newaxis]


def pair(n=1500, k=3, p1=2, p2=2, v=2, d=5, noise_rate=0.1, syn_type='ind'):
    if syn_type == 'ind':
        x = cause(n, k, p1, p2)
        return (x, scale(scale(mechanism(x, d)) + noise(n, v)))
    if syn_type == 'vio':
        x = cause(n, k, p1, p2)
        return (x,
                scale(scale(mechanism(x, d)) * (1 + noise_(n, np.sqrt(0.5)))))
    elif syn_type == 'dep':
        x = cause(n, k, p1, p2)
        return (x,
                scale(
                    scale(mechanism(x, d)) +
                    noise_rate * minmax_scale(x) * noise(n, v)))
    elif syn_type == 'hete':
        x = cause(n, k, p1, p2)
        return (x,
                scale(
                    scale(mechanism(x, d)) +
                    noise_rate * minmax_scale(x) * np.random.randn(n, 1)))
    else:
        raise TypeError


def generate(outdir,
             N,
             n_sample=1500,
             noise_rate=0.1,
             syn_type='ind',
             display=False):
    for i in range(N):
        x, y = pair(n=n_sample, noise_rate=noise_rate, syn_type=syn_type)
        if display:
            plt.figure()
            plt.scatter(x, y)
            check_dir(outdir + 'figs/')
            plt.savefig(outdir + 'figs/' + 'pairs%s.png' % (str(i).zfill(4)))
        df = pd.DataFrame({'x': x.reshape(-1), 'y': y.reshape(-1)})
        check_dir(outdir)
        df.to_csv(outdir + 'pairs%s.csv' % (str(i).zfill(4)), index=None)


def plot_datasets(files_path, fig_path):
    files = [
        os.path.join(files_path, name) for name in os.listdir(files_path)
        if os.path.isfile(files_path + name)
    ]
    _, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        x = df['x']
        y = df['y']
        # x = x.reshape(-1, 1)
        # y = y.reshape(-1, 1)
        axs[i // 10, i % 10].scatter(x, y, color='#3399ff')
        axs[i // 10, i % 10].axis('off')
    plt.savefig(fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Synthetic dataset for causal discovery.')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--syn_type',
                        type=str,
                        default='ind',
                        help='Synthetic type, ind, vio, dep, hete')
    parser.add_argument('--N',
                        type=int,
                        default=100,
                        help='Num of causal pair')
    parser.add_argument('--noise_rate',
                        type=float,
                        default=0.1,
                        help='Noise rate')
    parser.add_argument('--is_plot',
                        action='store_true',
                        default=False,
                        help='if plot all causal pair')
    args = parser.parse_args()
    np.random.seed(args.seed)

    syn_type = args.syn_type
    N = args.N
    noise_rate = args.noise_rate  # for syn_type = hete | vio
    is_plot = args.is_plot
    base_path = './'

    if syn_type in ['ind', 'dep']:
        outdir = base_path + f'datasets/synthetic_{syn_type}/'
        pdf_path = base_path + f'results/figs/synthetic_{syn_type}.pdf'
    else:
        outdir = base_path + f'datasets/synthetic_{syn_type}_{str(noise_rate)}/'
        pdf_path = base_path + f'results/figs/synthetic_{syn_type}_{str(noise_rate)}.pdf'
    check_dir(outdir)
    generate(outdir, N, noise_rate=noise_rate, syn_type=syn_type)
    if is_plot:
        plot_datasets(outdir, pdf_path)
