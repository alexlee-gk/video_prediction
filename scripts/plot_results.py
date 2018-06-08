from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os

import numpy as np


def load_metrics(prefix_fname):
    import csv
    with open('%s.csv' % prefix_fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        rows = list(reader)
        # skip header (first row), indices (first column), and means (last column)
        metrics = np.array(rows)[1:, 1:-1].astype(np.float32)
    return metrics


def plot_metric(metric, start_x=0, color=None, label=None, zorder=None):
    import matplotlib.pyplot as plt
    metric_mean = np.mean(metric, axis=0)
    metric_se = np.std(metric, axis=0) / np.sqrt(len(metric))
    kwargs = {}
    if color:
        kwargs['color'] = color
    if zorder:
        kwargs['zorder'] = zorder
    plt.errorbar(np.arange(len(metric_mean)) + start_x,
                 metric_mean, yerr=metric_se, linewidth=2,
                 label=label, **kwargs)
    # metric_std = np.std(metric, axis=0)
    # plt.plot(np.arange(len(metric_mean)) + start_x, metric_mean,
    #          linewidth=2, color=color, label=label)
    # plt.fill_between(np.arange(len(metric_mean)) + start_x,
    #                  metric_mean - metric_std, metric_mean + metric_std,
    #                  color=color, alpha=0.5)


def get_color(method_name):
    import matplotlib.pyplot as plt
    color_mapping = {
        'ours_vae_gan': plt.cm.Vega20(0),
        'ours_gan': plt.cm.Vega20(2),
        'ours_vae': plt.cm.Vega20(4),
        'ours_vae_l1': plt.cm.Vega20(4),
        'ours_vae_l2': plt.cm.Vega20(14),
        'ours_deterministic': plt.cm.Vega20(6),
        'ours_deterministic_l1': plt.cm.Vega20(6),
        'ours_deterministic_l2': plt.cm.Vega20(10),
        'sna_l1': plt.cm.Vega20(8),
        'sna_l2': plt.cm.Vega20(9),
        'sv2p_time_variant': plt.cm.Vega20(16),
        'sv2p_time_invariant': plt.cm.Vega20(16),
        'svg_lp': plt.cm.Vega20(18),
        'svg_fp': plt.cm.Vega20(18),
        'svg_fp_resized_data_loader': plt.cm.Vega20(18),
        'mathieu': plt.cm.Vega20(8),
        'mcnet': plt.cm.Vega20(8),
        'repeat': 'k',
    }
    if method_name in color_mapping:
        color = color_mapping[method_name]
    else:
        color = None
        for k, v in color_mapping.items():
            if method_name.startswith(k):
                color = v
                break
    return color


def get_method_name(method_name):
    method_name_mapping = {
        'ours_vae_gan': 'Ours, SAVP',
        'ours_gan': 'Ours, GAN-only',
        'ours_vae': 'Ours, VAE-only',
        'ours_vae_l1': 'Ours, VAE-only, $\mathcal{L}_1$',
        'ours_vae_l2': 'Ours, VAE-only, $\mathcal{L}_2$',
        'ours_deterministic': 'Ours, deterministic',
        'ours_deterministic_l1': 'Ours, deterministic, $\mathcal{L}_1$',
        'ours_deterministic_l2': 'Ours, deterministic, $\mathcal{L}_2$',
        'sna_l1': 'SNA, $\mathcal{L}_1$ (Ebert et al.)',
        'sna_l2': 'SNA, $\mathcal{L}_2$ (Ebert et al.)',
        'sv2p_time_variant': 'SV2P time-variant (Babaeizadeh et al.)',
        'sv2p_time_invariant': 'SV2P time-invariant (Babaeizadeh et al.)',
        'mathieu': 'Mathieu et al.',
        'mcnet': 'MCnet (Villegas et al.)',
        'repeat': 'Copy last frame',
    }
    return method_name_mapping.get(method_name, method_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--method_dirs", type=str, nargs='+', help='directories in results_dir (all of them by default)')
    parser.add_argument("--method_names", type=str, nargs='+', help='method names for the header')
    parser.add_argument("--web_dir", type=str, help='default is results_dir/web')
    parser.add_argument("--plot_fname", type=str, default='metrics.pdf')
    parser.add_argument('--usetex', '--use_tex', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--mode', choices=['paper', 'rebuttal'], default='paper')
    parser.add_argument("--plot_metric_names", type=str, nargs='+')
    args = parser.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt

    if args.usetex:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preview=True)
        plt.rc('font', family='serif')

    if args.web_dir is None:
        args.web_dir = os.path.join(args.results_dir, 'web')

    if args.method_dirs is None:
        unsorted_method_dirs = os.listdir(args.results_dir)
        # exclude web_dir and all directories that starts with web
        if args.web_dir in unsorted_method_dirs:
            unsorted_method_dirs.remove(args.web_dir)
        unsorted_method_dirs = [method_dir for method_dir in unsorted_method_dirs if not os.path.basename(method_dir).startswith('web')]
        # put ground_truth and repeat in the front (if any)
        method_dirs = []
        for first_method_dir in ['ground_truth', 'repeat']:
            if first_method_dir in unsorted_method_dirs:
                unsorted_method_dirs.remove(first_method_dir)
                method_dirs.append(first_method_dir)
        method_dirs.extend(sorted(unsorted_method_dirs))
    else:
        method_dirs = list(args.method_dirs)
    if args.method_names is None:
        method_names = [get_method_name(method_dir) for method_dir in method_dirs]
    else:
        method_names = list(args.method_names)
    if args.usetex:
        method_names = [method_name.replace('kl_weight', r'$\lambda_{\textsc{kl}}$') for method_name in method_names]
    method_dirs = [os.path.join(args.results_dir, method_dir) for method_dir in method_dirs]

    # infer task and metric names from first method
    metric_fnames = sorted(glob.glob('%s/*_max/metrics/*.csv' % glob.escape(method_dirs[0])))
    task_names = []
    metric_names = []  # all the metric names inferred from file names
    for metric_fname in metric_fnames:
        head, tail = os.path.split(metric_fname)
        task_name = head.split('/')[-2]
        metric_name, _ = os.path.splitext(tail)
        task_names.append(task_name)
        metric_names.append(metric_name)

    # save plots
    dataset_name = args.dataset_name or os.path.split(os.path.normpath(args.results_dir))[1]
    plots_dir = os.path.join(args.web_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if dataset_name in ('bair', 'bair_action_free'):
        context_frames = 2
        training_sequence_length = 12
        plot_metric_names = ('psnr', 'ssim_finn', 'vgg_csim')
    elif dataset_name == 'kth':
        context_frames = 10
        training_sequence_length = 20
        plot_metric_names = ('psnr', 'ssim_scikit', 'vgg_csim')
    elif dataset_name == 'ucf101':
        context_frames = 4
        training_sequence_length = 8
        plot_metric_names = ('psnr', 'ssim_mcnet', 'vgg_csim')
    else:
        raise NotImplementedError
    plot_metric_names = args.plot_metric_names or plot_metric_names  # metric names to plot

    if args.mode == 'paper':
        fig = plt.figure(figsize=(4 * len(plot_metric_names), 5))
    elif args.mode == 'rebuttal':
        fig = plt.figure(figsize=(4, 3 * len(plot_metric_names)))
    else:
        raise ValueError
    i_task = 0
    for task_name, metric_name in zip(task_names, metric_names):
        if not task_name.endswith('max'):
            continue
        if metric_name not in plot_metric_names:
            continue

        if args.mode == 'paper':
            plt.subplot(1, len(plot_metric_names), i_task + 1)
        elif args.mode == 'rebuttal':
            plt.subplot(len(plot_metric_names), 1, i_task + 1)

        for method_name, method_dir in zip(method_names, method_dirs):
            metric_fname = os.path.join(method_dir, task_name, 'metrics', metric_name)
            if not os.path.isfile('%s.csv' % metric_fname):
                print('Skipping', metric_fname)
                continue
            metric = load_metrics(metric_fname)
            plot_metric(metric, context_frames + 1, color=get_color(os.path.basename(method_dir)), label=method_name)

        plt.grid(axis='y')
        plt.axvline(x=training_sequence_length, linewidth=1, color='k')
        fontsize = 12 if args.mode == 'rebuttal' else 15
        legend_fontsize = 10 if args.mode == 'rebuttal' else 15
        labelsize = 10
        if args.mode == 'paper':
            plt.xlabel('Time Step', fontsize=fontsize)
        plt.ylabel({
            'psnr': 'Average PSNR',
            'ssim': 'Average SSIM',
            'ssim_scikit': 'Average SSIM',
            'ssim_finn': 'Average SSIM',
            'ssim_mcnet': 'Average SSIM',
            'vgg_csim': 'Average VGG cosine similarity',
        }[metric_name], fontsize=fontsize)
        plt.xlim((context_frames + 1, metric.shape[1] + context_frames))
        plt.tick_params(labelsize=labelsize)

        if args.mode == 'paper':
            if i_task == 1:
                # plt.title({
                #     'bair': 'Action-conditioned BAIR Dataset',
                #     'bair_action_free': 'Action-free BAIR Dataset',
                #     'kth': 'KTH Dataset',
                # }[dataset_name], fontsize=16)
                if len(method_names) <= 4 and sum([len(method_name) for method_name in method_names]) < 90:
                    ncol = len(method_names)
                else:
                    ncol = (len(method_names) + 1) // 2
                # ncol = 2
                plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=ncol, fontsize=legend_fontsize)
        elif args.mode == 'rebuttal':
            if i_task == 0:
                # plt.legend(fontsize=legend_fontsize)
                plt.legend(bbox_to_anchor=(0.4, -0.12), loc='upper center', fontsize=legend_fontsize)
            plt.ylim(ymin=0.8)
            plt.xlim((context_frames + 1, metric.shape[1] + context_frames))
        i_task += 1
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    if args.save:
        plt.show(block=False)
        print("Saving to", os.path.join(plots_dir, args.plot_fname))
        plt.savefig(os.path.join(plots_dir, args.plot_fname), bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()
