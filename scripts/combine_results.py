from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import itertools
import os

import cv2
import numpy as np

from video_prediction.utils import html
from video_prediction.utils.ffmpeg_gif import save_gif as ffmpeg_save_gif


def load_metrics(prefix_fname):
    import csv
    with open('%s.csv' % prefix_fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        rows = list(reader)
        # skip header (first row), indices (first column), and means (last column)
        metrics = np.array(rows)[1:, 1:-1].astype(np.float32)
    return metrics


def load_images(image_fnames):
    images = []
    for image_fname in image_fnames:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


def save_images(image_fnames, images):
    head, tail = os.path.split(image_fnames[0])
    if head and not os.path.exists(head):
        os.makedirs(head)
    for image_fname, image in zip(image_fnames, images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_fname, image)


def save_gif(gif_fname, images, fps=4):
    import moviepy.editor as mpy
    head, tail = os.path.split(gif_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
    clip.write_gif(gif_fname)


def concat_images(all_images):
    """
    all_images is a list of lists of images
    """
    min_height, min_width = None, None
    for all_image in all_images:
        for image in all_image:
            if min_height is None or min_width is None:
                min_height, min_width = image.shape[:2]
            else:
                min_height = min(min_height, image.shape[0])
                min_width = min(min_width, image.shape[1])

    def maybe_resize(image):
        if image.shape[:2] != (min_height, min_width):
            image = cv2.resize(image, (min_height, min_width))
        return image

    resized_all_images = []
    for all_image in all_images:
        resized_all_image = [maybe_resize(image) for image in all_image]
        resized_all_images.append(resized_all_image)
    all_images = resized_all_images
    all_images = [np.concatenate(all_image, axis=1) for all_image in zip(*all_images)]
    return all_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--method_dirs", type=str, nargs='+', help='directories in results_dir (all of them by default)')
    parser.add_argument("--method_names", type=str, nargs='+', help='method names for the header')
    parser.add_argument("--web_dir", type=str, help='default is results_dir/web')
    parser.add_argument("--sort_by", type=str, nargs=2, help='task and metric name to sort by, e.g. prediction mse')
    parser.add_argument("--no_ffmpeg", action='store_true')
    parser.add_argument("--batch_size", type=int, default=1, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples for the table of sequence (all of them by default)")
    parser.add_argument("--show_se", action='store_true', help="show standard error in the table metrics")
    parser.add_argument("--only_metrics", action='store_true')
    args = parser.parse_args()

    if args.web_dir is None:
        args.web_dir = os.path.join(args.results_dir, 'web')
    webpage = html.HTML(args.web_dir, 'Experiment name = %s' % os.path.normpath(args.results_dir), reflesh=1)
    webpage.add_header1(os.path.normpath(args.results_dir))

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
        method_names = list(method_dirs)
    else:
        method_names = list(args.method_names)
    method_dirs = [os.path.join(args.results_dir, method_dir) for method_dir in method_dirs]

    if args.sort_by:
        task_name, metric_name = args.sort_by
        sort_criterion = []
        for method_id, (method_name, method_dir) in enumerate(zip(method_names, method_dirs)):
            metric = load_metrics(os.path.join(method_dir, task_name, 'metrics', metric_name))
            sort_criterion.append(np.mean(metric))
        sort_criterion, method_ids, method_names, method_dirs = \
            zip(*sorted(zip(sort_criterion, range(len(method_names)), method_names, method_dirs)))
        webpage.add_header3('sorted by %s, %s' % tuple(args.sort_by))
    else:
        method_ids = range(len(method_names))

    # infer task and metric names from first method
    metric_fnames = sorted(glob.glob('%s/*/metrics/*.csv' % glob.escape(method_dirs[0])))
    task_names = []
    metric_names = []
    for metric_fname in metric_fnames:
        head, tail = os.path.split(metric_fname)
        task_name = head.split('/')[-2]
        metric_name, _ = os.path.splitext(tail)
        task_names.append(task_name)
        metric_names.append(metric_name)

    # save metrics
    webpage.add_table()
    header_txts = ['']
    header_colspans = [2]
    for task_name in task_names:
        if task_name != header_txts[-1]:
            header_txts.append(task_name)
            header_colspans.append(2 if args.show_se else 1)  # mean and standard error for each task
        else:
            # group consecutive task names that are the same
            header_colspans[-1] += 2 if args.show_se else 1
    webpage.add_row(header_txts, header_colspans)
    subheader_txts = ['id', 'method']
    for task_name, metric_name in zip(task_names, metric_names):
        subheader_txts.append('%s (mean)' % metric_name)
        if args.show_se:
            subheader_txts.append('%s (se)' % metric_name)
    webpage.add_row(subheader_txts)
    all_metric_means = []
    for method_id, method_name, method_dir in zip(method_ids, method_names, method_dirs):
        metric_txts = [method_id, method_name]
        metric_means = []
        for task_name, metric_name in zip(task_names, metric_names):
            metric = load_metrics(os.path.join(method_dir, task_name, 'metrics', metric_name))
            metric_mean = np.mean(metric)
            num_samples = len(metric)
            metric_se = np.std(metric) / np.sqrt(num_samples)
            metric_txts.append('%.4f' % metric_mean)
            if args.show_se:
                metric_txts.append('%.4f' % metric_se)
            metric_means.append(metric_mean)
        webpage.add_row(metric_txts)
        all_metric_means.append(metric_means)
    webpage.save()

    if args.only_metrics:
        return

    # infer task names from first method
    outputs_dirs = sorted(glob.glob('%s/*/outputs' % glob.escape(method_dirs[0])))
    task_names = [outputs_dir.split('/')[-2] for outputs_dir in outputs_dirs]

    # save image sequences
    image_dir = os.path.join(args.web_dir, 'images')
    webpage.add_table()
    header_txts = ['']
    subheader_txts = ['id']
    methods_subheader_txts = ['']
    header_colspans = [1]
    subheader_colspans = [1]
    methods_subheader_colspans = [1]
    num_samples = args.num_samples or num_samples
    for sample_ind in range(num_samples):
        if sample_ind % args.batch_size == 0:
            print("saving samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))
        ims = [None]
        txts = [sample_ind]
        links = [None]
        colspans = [1]
        for task_name in task_names:
            # load input images from first method
            input_fnames = sorted(glob.glob('%s/inputs/*_%05d_??.png' %
                                            (glob.escape(os.path.join(method_dirs[0], task_name)), sample_ind)))
            input_images = load_images(input_fnames)
            # save input images as image sequence
            input_fnames = [os.path.join(task_name, 'inputs', os.path.basename(input_fname)) for input_fname in input_fnames]
            save_images([os.path.join(image_dir, input_fname) for input_fname in input_fnames], input_images)
            # infer output names from first method
            output_fnames = sorted(glob.glob('%s/outputs/*_%05d_??.png' %
                                             (glob.escape(os.path.join(method_dirs[0], task_name)), sample_ind)))
            output_names = sorted(set(os.path.splitext(os.path.basename(output_fname))[0][:-9]
                                      for output_fname in output_fnames))  # remove _?????_??.png
            # load output images
            all_output_images = []
            for output_name in output_names:
                for method_name, method_dir in zip(method_names, method_dirs):
                    output_fnames = sorted(glob.glob('%s/outputs/%s_%05d_??.png' %
                                                     (glob.escape(os.path.join(method_dir, task_name)),
                                                      output_name, sample_ind)))
                    output_images = load_images(output_fnames)
                    all_output_images.append(output_images)
            # concatenate output images of all the methods
            all_output_images = concat_images(all_output_images)
            # save output images as image sequence or as gif clip
            output_fname = os.path.join(task_name, 'outputs', '%s_%05d.gif' % ('_'.join(output_names), sample_ind))
            if args.no_ffmpeg:
                save_gif(os.path.join(image_dir, output_fname), all_output_images, fps=4)
            else:
                ffmpeg_save_gif(os.path.join(image_dir, output_fname), all_output_images, fps=4)

            if sample_ind == 0:
                header_txts.append(task_name)
                subheader_txts.extend(['inputs', 'outputs'])
                header_colspans.append(len(input_fnames) + len(method_ids) * len(output_names))
                subheader_colspans.extend([len(input_fnames), len(method_ids) * len(output_names)])
                method_id_strs = ['%02d' % method_id for method_id in method_ids]
                methods_subheader_txts.extend([''] + list(itertools.chain(*[method_id_strs] * len(output_names))))
                methods_subheader_colspans.extend([len(input_fnames)] + [1] * (len(method_ids) * len(output_names)))
            ims.extend(input_fnames + [output_fname])
            txts.extend([None] * (len(input_fnames) + 1))
            links.extend(input_fnames + [output_fname])
            colspans.extend([1] * len(input_fnames) + [len(method_ids) * len(output_names)])

        if sample_ind == 0:
            webpage.add_row(header_txts, header_colspans)
            webpage.add_row(subheader_txts, subheader_colspans)
            webpage.add_row(methods_subheader_txts, methods_subheader_colspans)
        webpage.add_images(ims, txts, links, colspans, height=64, width=None)
        if (sample_ind + 1) % args.batch_size == 0:
            webpage.save()
    webpage.save()


if __name__ == '__main__':
    main()
