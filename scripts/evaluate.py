import argparse
import csv
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import datasets, models, metrics
from video_prediction.policies.servo_policy import ServoPolicy


def compute_expectation_np(pix_distrib):
    assert pix_distrib.shape[-1] == 1
    pix_distrib = pix_distrib / np.sum(pix_distrib, axis=(-3, -2), keepdims=True)
    height, width = pix_distrib.shape[-3:-1]
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([np.sum(yv[:, :, None] * pix_distrib, axis=(-3, -2, -1)),
                     np.sum(xv[:, :, None] * pix_distrib, axis=(-3, -2, -1))], axis=-1)


def as_heatmap(image, normalize=True):
    import matplotlib.pyplot as plt
    image = np.squeeze(image, axis=-1)
    if normalize:
        image = image / np.max(image, axis=(-2, -1), keepdims=True)
    cmap = plt.get_cmap('viridis')
    heatmap = cmap(image)[..., :3]
    return heatmap


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def resize_and_draw_circle(image, size, center, radius, dpi=128.0, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import io
    height, width = size
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation='none')
    circle = Circle(center[::-1], radius=radius, **kwargs)
    ax.add_patch(circle)
    ax.axis("off")
    fig.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox)
    buff.seek(0)
    image = plt.imread(buff)[..., :3]
    plt.close(fig)
    return image


def save_image_sequence(prefix_fname, images, overlaid_images=None, centers=None,
                        radius=5, alpha=0.8, time_start_ind=0):
    import cv2
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head, exist_ok=True)
    if images.shape[-1] == 1:
        images = as_heatmap(images)
    if overlaid_images is not None:
        assert images.shape[-1] == 3
        assert overlaid_images.shape[-1] == 1
        gray_images = rgb2gray(images)
        overlaid_images = as_heatmap(overlaid_images)
        images = (1 - alpha) * gray_images[..., None] + alpha * overlaid_images
    for t, image in enumerate(images):
        image_fname = '%s_%02d.jpg' % (prefix_fname, time_start_ind + t)
        if centers is not None:
            scale = np.max(np.array([256, 256]) / np.array(image.shape[:2]))
            image = resize_and_draw_circle(image, np.array(image.shape[:2]) * scale, centers[t], radius,
                                           edgecolor='r', fill=False, linestyle='--', linewidth=2)
        image = (image * 255.0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_fname, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def save_image_sequences(prefix_fname, images, overlaid_images=None, centers=None,
                         radius=5, alpha=0.8, sample_start_ind=0, time_start_ind=0):
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head, exist_ok=True)
    if overlaid_images is None:
        overlaid_images = [None] * len(images)
    if centers is None:
        centers = [None] * len(images)
    for i, (images_, overlaid_images_, centers_) in enumerate(zip(images, overlaid_images, centers)):
        images_fname = '%s_%05d' % (prefix_fname, sample_start_ind + i)
        save_image_sequence(images_fname, images_, overlaid_images_, centers_,
                            radius=radius, alpha=alpha, time_start_ind=time_start_ind)


def save_metrics(prefix_fname, metrics, sample_start_ind=0):
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head, exist_ok=True)
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


def save_prediction_results(task_dir, results, model_hparams, sample_start_ind=0, only_metrics=False):
    context_frames = model_hparams.context_frames
    sequence_length = model_hparams.sequence_length
    context_images, images = np.split(results['images'], [context_frames], axis=1)
    gen_images = results['gen_images'][:, context_frames - sequence_length:]
    psnr = metrics.peak_signal_to_noise_ratio_np(images, gen_images, axis=tuple(range(2, images.ndim)))
    mse = metrics.mean_squared_error_np(images, gen_images, axis=tuple(range(2, images.ndim)))
    ssim = metrics.structural_similarity_np(images, gen_images, axis=())
    save_metrics(os.path.join(task_dir, 'metrics', 'psnr'),
                 psnr, sample_start_ind=sample_start_ind)
    save_metrics(os.path.join(task_dir, 'metrics', 'mse'),
                 mse, sample_start_ind=sample_start_ind)
    save_metrics(os.path.join(task_dir, 'metrics', 'ssim'),
                 ssim, sample_start_ind=sample_start_ind)
    if only_metrics:
        return

    save_image_sequences(os.path.join(task_dir, 'inputs', 'context_image'),
                         context_images, sample_start_ind=sample_start_ind)
    save_image_sequences(os.path.join(task_dir, 'outputs', 'gen_image'),
                         gen_images, sample_start_ind=sample_start_ind)


def save_motion_results(task_dir, results, model_hparams, draw_center=False,
                        sample_start_ind=0, only_metrics=False):
    context_frames = model_hparams.context_frames
    sequence_length = model_hparams.sequence_length
    pix_distribs = results['pix_distribs'][:, context_frames:]
    gen_pix_distribs = results['gen_pix_distribs'][:, context_frames - sequence_length:]
    pix_dist = metrics.expected_pixel_distance_np(pix_distribs, gen_pix_distribs, axis=-1)
    save_metrics(os.path.join(task_dir, 'metrics', 'pix_dist'),
                 pix_dist, sample_start_ind=sample_start_ind)
    if only_metrics:
        return

    context_images, images = np.split(results['images'], [context_frames], axis=1)
    gen_images = results['gen_images'][:, context_frames - sequence_length:]
    initial_pix_distrib = results['pix_distribs'][:, 0:1]
    save_image_sequences(os.path.join(task_dir, 'inputs', 'pix_distrib'),
                         initial_pix_distrib, sample_start_ind=sample_start_ind)
    save_image_sequences(os.path.join(task_dir, 'outputs', 'gen_pix_distrib'),
                         gen_pix_distribs, sample_start_ind=sample_start_ind)

    centers = compute_expectation_np(initial_pix_distrib) if draw_center else None
    save_image_sequences(os.path.join(task_dir, 'inputs', 'pix_distrib'),
                         context_images[:, 0:1], initial_pix_distrib, centers, sample_start_ind=sample_start_ind)
    centers = compute_expectation_np(gen_pix_distribs) if draw_center else None
    save_image_sequences(os.path.join(task_dir, 'outputs', 'gen_pix_distrib_overlaid'),
                         gen_images, gen_pix_distribs, centers, sample_start_ind=sample_start_ind)


def save_servo_results(task_dir, results, model_hparams, sample_start_ind=0, only_metrics=False):
    context_frames = model_hparams.context_frames
    sequence_length = model_hparams.sequence_length
    context_images, images = np.split(results['images'], [context_frames], axis=1)
    gen_images = results['gen_images'][:, context_frames - sequence_length:]
    goal_image = results['goal_image']
    # TODO: should exclude "context" actions assuming that they are passed in to the network
    actions = results['actions']
    gen_actions = results['gen_actions']
    goal_image_mse = metrics.mean_squared_error_np(goal_image, gen_images[:, -1], axis=(1, 2, 3))
    action_mse = metrics.mean_squared_error_np(actions, gen_actions, axis=tuple(range(2, actions.ndim)))
    save_metrics(os.path.join(task_dir, 'metrics', 'goal_image_mse'),
                 goal_image_mse[:, None], sample_start_ind=sample_start_ind)
    save_metrics(os.path.join(task_dir, 'metrics', 'action_mse'),
                 action_mse, sample_start_ind=sample_start_ind)
    if only_metrics:
        return

    save_image_sequences(os.path.join(task_dir, 'inputs', 'context_image'),
                         context_images, sample_start_ind=sample_start_ind)
    save_image_sequences(os.path.join(task_dir, 'inputs', 'goal_image'),
                         goal_image[:, None], sample_start_ind=sample_start_ind)
    save_image_sequences(os.path.join(task_dir, 'outputs', 'gen_image'),
                         gen_images, sample_start_ind=sample_start_ind)
    gen_image_goal_diffs = np.abs(gen_images - goal_image[:, None])
    save_image_sequences(os.path.join(task_dir, 'outputs', 'gen_image_goal_diff'),
                         gen_image_goal_diffs, sample_start_ind=sample_start_ind)


def main():
    """
    output_dir                              # condition / method
    ├── prediction                          # task
    │   ├── inputs
    │   │   ├── context_image_00000_00.jpg  # indexed by sample index and time step
    │   │   └── ...
    │   ├── outputs
    │   │   ├── gen_image_00000_00.jpg      # predicted images (only the ones in the loss)
    │   │   └── ...
    │   └── metrics
    │       ├── psnr.csv
    │       ├── mse.csv
    │       └── ssim.csv
    ├── servo
    │   ├── inputs
    │   │   ├── context_image_00000_00.jpg
    │   │   ├── ...
    │   │   ├── goal_image_00000_00.jpg     # only one goal image per sample
    │   │   └── ...
    │   ├── outputs
    │   │   ├── gen_image_00000_00.jpg
    │   │   ├── ...
    │   │   ├── gen_image_goal_diff_00000_00.jpg
    │   │   └── ...
    │   └── metrics
    │       ├── action_mse.csv
    │       └── goal_image_mse.csv
    └── motion
        ├── inputs
        │   ├── pix_distrib_00000_00.jpg
        │   └── ...
        ├── outputs
        │   ├── gen_pix_distrib_00000_00.jpg
        │   ├── ...
        │   ├── gen_pix_distrib_overlaid_00000_00.jpg
        │   └── ...
        └── metrics
            └── pix_dist.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val')
    parser.add_argument("--input_dir", type=str, required=True,
                        help="either a directory containing subdirectories train,"
                             "val, test, etc, or a directory containing the tfrecords")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=str,
                        help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000) "
                             "to resume training from or use for testing. Can specify multiple checkpoints. "
                             "If more than one checkpoint is provided, the global step from the checkpoints "
                             "are not restored.")
    parser.add_argument("--batch_size", type=int, default=16, help="number of samples in batch")
    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--tasks", type=str, nargs='+', help='tasks to evaluation (e.g. prediction, servo, motion)')
    parser.add_argument("--only_metrics", action='store_true')
    parser.add_argument("--num_stochastic_samples", type=int, default=0)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            checkpoint_dir = args.checkpoint
        else:
            checkpoint_dir, _ = os.path.split(args.checkpoint)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.dataset_hparams = merge_hparams(options['dataset_hparams'], args.dataset_hparams)
            args.model = args.model or options['model']
            args.model_hparams = merge_hparams(options['model_hparams'], args.model_hparams)
        output_dir = os.path.join(args.results_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        output_dir = os.path.join(args.results_dir, args.model)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(args.input_dir, mode=args.mode, num_epochs=1, hparams=args.dataset_hparams)
    inputs, _ = dataset.make_batch(args.batch_size)

    tasks = args.tasks
    if tasks is None:
        tasks = ['prediction', 'servo']
        if 'pix_distribs' in inputs:
            tasks.append('motion')

    VideoPredictionModel = models.get_model_class(args.model)
    model_hparams_dict = dict(context_frames=dataset.hparams.context_frames,
                              sequence_length=dataset.hparams.sequence_length,
                              repeat=dataset.hparams.time_shift)
    model = VideoPredictionModel(mode=args.mode, hparams_dict=model_hparams_dict, hparams=args.model_hparams)
    context_frames = model.hparams.context_frames
    sequence_length = model.hparams.sequence_length

    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
    with tf.variable_scope(''):
        model.build_graph(input_phs)
    if 'servo' in tasks:
        servo_model = VideoPredictionModel(mode=args.mode, hparams_dict=model_hparams_dict, hparams=args.model_hparams)
        cem_batch_size = 200
        plan_horizon = sequence_length - 1
        image_shape = inputs['images'].shape.as_list()[2:]
        state_shape = inputs['states'].shape.as_list()[2:]
        action_shape = inputs['actions'].shape.as_list()[2:]
        servo_input_phs = {
            'images': tf.placeholder(tf.float32, shape=[cem_batch_size, context_frames] + image_shape),
            'states': tf.placeholder(tf.float32, shape=[cem_batch_size, 1] + state_shape),
            'actions': tf.placeholder(tf.float32, shape=[cem_batch_size, plan_horizon] + action_shape),
        }
        if isinstance(servo_model, models.GroundTruthVideoPredictionModel):
            images_shape = inputs['images'].shape.as_list()[1:]
            servo_input_phs['images'] = tf.placeholder(tf.float32, shape=[cem_batch_size] + images_shape)
        with tf.variable_scope('', reuse=True):
            servo_model.build_graph(servo_input_phs)

    if not isinstance(model, models.NonTrainableVideoPredictionModel):
        model.build_restore_graph(args.checkpoint)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    if not isinstance(model, models.NonTrainableVideoPredictionModel):
        sess.run(model.restore_op)

    if 'servo' in tasks:
        servo_policy = ServoPolicy(servo_model, sess)

    sample_ind = 0
    while True:
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))

        if 'prediction' in tasks or 'motion' in tasks:  # do these together
            feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
            fetches = {'images': model.inputs['images'],
                       'gen_images': model.outputs['gen_images']}
            if 'motion' in tasks:
                fetches.update({'pix_distribs': model.inputs['pix_distribs'],
                                'gen_pix_distribs': model.outputs['gen_pix_distribs']})

            if args.num_stochastic_samples:
                all_results = [sess.run(fetches, feed_dict=feed_dict) for _ in range(args.num_stochastic_samples)]
                all_results = nest.map_structure(lambda *x: np.stack(x), *all_results)
                all_context_images, all_images = np.split(all_results['images'], [context_frames], axis=2)
                all_gen_images = all_results['gen_images'][:, :, context_frames - sequence_length:]
                all_mse = metrics.mean_squared_error_np(all_images, all_gen_images, axis=tuple(range(2, all_images.ndim)))
                all_mse_argsort = np.argsort(all_mse, axis=0)

                for subtask, argsort_ind in zip(['_best', '_median', '_worst'],
                                                [0, args.num_stochastic_samples // 2, -1]):
                    all_mse_inds = all_mse_argsort[argsort_ind]
                    gather = lambda x: np.array([x[ind, sample_ind] for sample_ind, ind in enumerate(all_mse_inds)])
                    results = nest.map_structure(gather, all_results)
                    if 'prediction' in tasks:
                        save_prediction_results(os.path.join(output_dir, 'prediction' + subtask),
                                                results, model.hparams, sample_ind, args.only_metrics)
                    if 'motion' in tasks:
                        draw_center = isinstance(model, models.NonTrainableVideoPredictionModel),
                        save_motion_results(os.path.join(output_dir, 'motion' + subtask),
                                            results, model.hparams, draw_center, sample_ind, args.only_metrics)
            else:
                results = sess.run(fetches, feed_dict=feed_dict)
                if 'prediction' in tasks:
                    save_prediction_results(os.path.join(output_dir, 'prediction'),
                                            results, model.hparams, sample_ind, args.only_metrics)
                if 'motion' in tasks:
                    draw_center = isinstance(model, models.NonTrainableVideoPredictionModel),
                    save_motion_results(os.path.join(output_dir, 'motion'),
                                        results, model.hparams, draw_center, sample_ind, args.only_metrics)

        if 'servo' in tasks:
            images = input_results['images']
            states = input_results['states']
            gen_actions = []
            gen_images = []
            for images_, states_ in zip(images, states):
                obs = {'context_images': images_[:context_frames],
                       'context_state': states_[0],
                       'goal_image': images_[-1]}
                if isinstance(servo_model, models.GroundTruthVideoPredictionModel):
                    obs['context_images'] = images_
                gen_actions_, gen_images_ = servo_policy.act(obs, servo_model.outputs['gen_images'])
                gen_actions.append(gen_actions_)
                gen_images.append(gen_images_)
            gen_actions = np.stack(gen_actions)
            gen_images = np.stack(gen_images)
            results = {'images': input_results['images'],
                       'actions': input_results['actions'],
                       'goal_image': input_results['images'][:, -1],
                       'gen_actions': gen_actions,
                       'gen_images': gen_images}
            save_servo_results(os.path.join(output_dir, 'servo'),
                               results, servo_model.hparams, sample_ind, args.only_metrics)

        sample_ind += args.batch_size

    metric_fnames = []
    if 'prediction' in tasks:
        subtask = '_best' if args.num_stochastic_samples else ''
        metric_fnames.extend([
            os.path.join(output_dir, 'prediction' + subtask, 'metrics', 'psnr'),
            os.path.join(output_dir, 'prediction' + subtask, 'metrics', 'mse'),
            os.path.join(output_dir, 'prediction' + subtask, 'metrics', 'ssim'),
        ])
    if 'motion' in tasks:
        subtask = '_best' if args.num_stochastic_samples else ''
        metric_fnames.append(os.path.join(output_dir, 'motion' + subtask, 'metrics', 'pix_dist'))
    if 'servo' in tasks:
        metric_fnames.append(os.path.join(output_dir, 'servo', 'metrics', 'goal_image_mse'))
        metric_fnames.append(os.path.join(output_dir, 'servo', 'metrics', 'action_mse'))

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
