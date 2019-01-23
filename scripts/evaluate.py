from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import csv
import errno
import json
import os
import random

import numpy as np
import tensorflow as tf

from video_prediction import datasets, models


def save_image_sequence(prefix_fname, images, time_start_ind=0):
    import cv2
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    for t, image in enumerate(images):
        image_fname = '%s_%02d.png' % (prefix_fname, time_start_ind + t)
        image = (image * 255.0).astype(np.uint8)
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_fname, image)


def save_image_sequences(prefix_fname, images, sample_start_ind=0, time_start_ind=0):
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    for i, images_ in enumerate(images):
        images_fname = '%s_%05d' % (prefix_fname, sample_start_ind + i)
        save_image_sequence(images_fname, images_, time_start_ind=time_start_ind)


def save_metrics(prefix_fname, metrics, sample_start_ind=0):
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    assert metrics.ndim == 2
    file_mode = 'w' if sample_start_ind == 0 else 'a'
    with open('%s.csv' % prefix_fname, file_mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if sample_start_ind == 0:
            writer.writerow(map(str, ['sample_ind'] + list(range(metrics.shape[1])) + ['mean']))
        for i, metrics_row in enumerate(metrics):
            writer.writerow(map(str, [sample_start_ind + i] + list(metrics_row) + [np.mean(metrics_row)]))


def load_metrics(prefix_fname):
    with open('%s.csv' % prefix_fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        rows = list(reader)
        # skip header (first row), indices (first column), and means (last column)
        metrics = np.array(rows)[1:, 1:-1].astype(np.float32)
    return metrics


def merge_hparams(hparams0, hparams1):
    hparams0 = hparams0 or []
    hparams1 = hparams1 or []
    if not isinstance(hparams0, (list, tuple)):
        hparams0 = [hparams0]
    if not isinstance(hparams1, (list, tuple)):
        hparams1 = [hparams1]
    hparams = list(hparams0) + list(hparams1)
    # simplify into the content if possible
    if len(hparams) == 1:
        hparams, = hparams
    return hparams


def save_prediction_eval_results(task_dir, results, model_hparams, sample_start_ind=0, only_metrics=False, subtasks=None):
    sequence_length = model_hparams.sequence_length
    context_frames = model_hparams.context_frames
    future_length = sequence_length - context_frames

    context_images = results['images'][:, :context_frames]

    if 'eval_diversity' in results:
        metric = results['eval_diversity']
        metric_name = 'diversity'
        subtask_dir = task_dir + '_%s' % metric_name
        save_metrics(os.path.join(subtask_dir, 'metrics', metric_name),
                     metric, sample_start_ind=sample_start_ind)

    subtasks = subtasks or ['max']
    for subtask in subtasks:
        metric_names = []
        for k in results.keys():
            if re.match('eval_(\w+)/%s' % subtask, k) and not re.match('eval_gen_images_(\w+)/%s' % subtask, k):
                m = re.match('eval_(\w+)/%s' % subtask, k)
                metric_names.append(m.group(1))
        for metric_name in metric_names:
            subtask_dir = task_dir + '_%s_%s' % (metric_name, subtask)
            gen_images = results.get('eval_gen_images_%s/%s' % (metric_name, subtask), results.get('eval_gen_images'))
            # only keep the future frames
            gen_images = gen_images[:, -future_length:]
            metric = results['eval_%s/%s' % (metric_name, subtask)]
            save_metrics(os.path.join(subtask_dir, 'metrics', metric_name),
                         metric, sample_start_ind=sample_start_ind)
            if only_metrics:
                continue

            save_image_sequences(os.path.join(subtask_dir, 'inputs', 'context_image'),
                                 context_images, sample_start_ind=sample_start_ind)
            save_image_sequences(os.path.join(subtask_dir, 'outputs', 'gen_image'),
                                 gen_images, sample_start_ind=sample_start_ind)


def main():
    """
    results_dir
    ├── output_dir                              # condition / method
    │   ├── prediction_eval_lpips_max           # task: best sample in terms of LPIPS similarity
    │   │   ├── inputs
    │   │   │   ├── context_image_00000_00.png  # indexed by sample index and time step
    │   │   │   └── ...
    │   │   ├── outputs
    │   │   │   ├── gen_image_00000_00.png      # predicted images (only the future ones)
    │   │   │   └── ...
    │   │   └── metrics
    │   │       └── lpips.csv
    │   ├── prediction_eval_ssim_max            # task: best sample in terms of SSIM
    │   │   ├── inputs
    │   │   │   ├── context_image_00000_00.png  # indexed by sample index and time step
    │   │   │   └── ...
    │   │   ├── outputs
    │   │   │   ├── gen_image_00000_00.png      # predicted images (only the future ones)
    │   │   │   └── ...
    │   │   └── metrics
    │   │       └── ssim.csv
    │   └── ...
    └── ...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where results are saved. default is results_dir/model_fname, "
                                             "where model_fname is the directory name of checkpoint")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")

    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help='mode for dataset, val or test.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")

    parser.add_argument("--batch_size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--eval_substasks", type=str, nargs='+', default=['max', 'avg', 'min'], help='subtasks to evaluate (e.g. max, avg, min)')
    parser.add_argument("--only_metrics", action='store_true')
    parser.add_argument("--num_stochastic_samples", type=int, default=100)

    parser.add_argument("--gt_inputs_dir", type=str, help="directory containing input ground truth images for ismple dataset")
    parser.add_argument("--gt_outputs_dir", type=str, help="directory containing output ground truth images for ismple dataset")

    parser.add_argument("--eval_parallel_iterations", type=int, default=10)
    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
        args.output_dir = args.output_dir or os.path.join(args.results_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_dir = args.output_dir or os.path.join(args.results_dir, 'model.%s' % args.model)

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(
        args.input_dir,
        mode=args.mode,
        num_epochs=args.num_epochs,
        seed=args.seed,
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)

    VideoPredictionModel = models.get_model_class(args.model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': dataset.hparams.context_frames,
        'sequence_length': dataset.hparams.sequence_length,
        'repeat': dataset.hparams.time_shift,
    })
    model = VideoPredictionModel(
        mode=args.mode,
        hparams_dict=hparams_dict,
        hparams=args.model_hparams,
        eval_num_samples=args.num_stochastic_samples,
        eval_parallel_iterations=args.eval_parallel_iterations)

    if args.num_samples:
        if args.num_samples > dataset.num_examples_per_epoch():
            raise ValueError('num_samples cannot be larger than the dataset')
        num_examples_per_epoch = args.num_samples
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch()
    if num_examples_per_epoch % args.batch_size != 0:
        raise ValueError('batch_size should evenly divide the dataset size %d' % num_examples_per_epoch)

    inputs = dataset.make_batch(args.batch_size)
    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
    with tf.variable_scope(''):
        model.build_graph(input_phs)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    sample_ind = 0
    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))

        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        # compute "best" metrics using the computation graph
        fetches = {'images': model.inputs['images']}
        fetches.update(model.eval_outputs.items())
        fetches.update(model.eval_metrics.items())
        results = sess.run(fetches, feed_dict=feed_dict)
        save_prediction_eval_results(os.path.join(output_dir, 'prediction_eval'),
                                     results, model.hparams, sample_ind, args.only_metrics, args.eval_substasks)
        sample_ind += args.batch_size

    metric_fnames = []
    metric_names = ['psnr', 'ssim', 'lpips']
    subtasks = ['max']
    for metric_name in metric_names:
        for subtask in subtasks:
            metric_fnames.append(
                os.path.join(output_dir, 'prediction_eval_%s_%s' % (metric_name, subtask), 'metrics', metric_name))

    for metric_fname in metric_fnames:
        task_name, _, metric_name = metric_fname.split('/')[-3:]
        metric = load_metrics(metric_fname)
        print('=' * 31)
        print(task_name, metric_name)
        print('-' * 31)
        metric_header_format = '{:>10} {:>20}'
        metric_row_format = '{:>10} {:>10.4f} ({:>7.4f})'
        print(metric_header_format.format('time step', os.path.split(metric_fname)[1]))
        for t, (metric_mean, metric_std) in enumerate(zip(metric.mean(axis=0), metric.std(axis=0))):
            print(metric_row_format.format(t, metric_mean, metric_std))
        print(metric_row_format.format('mean (std)', metric.mean(), metric.std()))
        print('=' * 31)


if __name__ == '__main__':
    main()
