from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random
import time

import numpy as np
import tensorflow as tf

from video_prediction import datasets, models


def add_tag_suffix(summary, tag_suffix):
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary)
    summary = summary_proto

    for value in summary.value:
        tag_split = value.tag.split('/')
        value.tag = '/'.join([tag_split[0] + tag_suffix] + tag_split[1:])
    return summary.SerializeToString()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--val_input_dir", type=str, help="directories containing the tfrecords. default: input_dir")
    parser.add_argument("--logs_dir", default='logs', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--output_dir_postfix", default="")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--resume", action='store_true', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--dataset_hparams_dict", type=str, help="a json file of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")

    parser.add_argument("--summary_freq", type=int, default=1000, help="save frequency of summaries (except for image and eval summaries) for train/validation set")
    parser.add_argument("--image_summary_freq", type=int, default=5000, help="save frequency of image summaries for train/validation set")
    parser.add_argument("--eval_summary_freq", type=int, default=25000, help="save frequency of eval summaries for train/validation set")
    parser.add_argument("--accum_eval_summary_freq", type=int, default=100000, help="save frequency of accumulated eval summaries for validation set only")
    parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save frequence of model, 0 to disable")

    parser.add_argument("--aggregate_nccl", type=int, default=0, help="whether to use nccl or cpu for gradient aggregation in multi-gpu training")
    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.output_dir is None:
        list_depth = 0
        model_fname = ''
        for t in ('model=%s,%s' % (args.model, args.model_hparams)):
            if t == '[':
                list_depth += 1
            if t == ']':
                list_depth -= 1
            if list_depth and t == ',':
                t = '..'
            if t in '=,':
                t = '.'
            if t in '[]':
                t = ''
            model_fname += t
        args.output_dir = os.path.join(args.logs_dir, model_fname) + args.output_dir_postfix

    if args.resume:
        if args.checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        args.checkpoint = args.output_dir

    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.dataset_hparams_dict:
        with open(args.dataset_hparams_dict) as f:
            dataset_hparams_dict.update(json.loads(f.read()))
    if args.model_hparams_dict:
        with open(args.model_hparams_dict) as f:
            model_hparams_dict.update(json.loads(f.read()))
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
                dataset_hparams_dict.update(json.loads(f.read()))
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict.update(json.loads(f.read()))
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    train_dataset = VideoDataset(
        args.input_dir,
        mode='train',
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)
    val_dataset = VideoDataset(
        args.val_input_dir or args.input_dir,
        mode='val',
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)
    if val_dataset.hparams.long_sequence_length != val_dataset.hparams.sequence_length:
        # the longer dataset is only used for the accum_eval_metrics
        long_val_dataset = VideoDataset(
            args.val_input_dir or args.input_dir,
            mode='val',
            hparams_dict=dataset_hparams_dict,
            hparams=args.dataset_hparams)
        long_val_dataset.set_sequence_length(val_dataset.hparams.long_sequence_length)
    else:
        long_val_dataset = None

    variable_scope = tf.get_variable_scope()
    variable_scope.set_use_resource(True)

    VideoPredictionModel = models.get_model_class(args.model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': train_dataset.hparams.context_frames,
        'sequence_length': train_dataset.hparams.sequence_length,
        'repeat': train_dataset.hparams.time_shift,
    })
    model = VideoPredictionModel(
        hparams_dict=hparams_dict,
        hparams=args.model_hparams,
        aggregate_nccl=args.aggregate_nccl)

    batch_size = model.hparams.batch_size
    train_tf_dataset = train_dataset.make_dataset(batch_size)
    train_iterator = train_tf_dataset.make_one_shot_iterator()
    train_handle = train_iterator.string_handle()
    val_tf_dataset = val_dataset.make_dataset(batch_size)
    val_iterator = val_tf_dataset.make_one_shot_iterator()
    val_handle = val_iterator.string_handle()
    iterator = tf.data.Iterator.from_string_handle(
        train_handle, train_tf_dataset.output_types, train_tf_dataset.output_shapes)
    inputs = iterator.get_next()

    # inputs comes from the training dataset by default, unless train_handle is remapped to the val_handles
    model.build_graph(inputs)

    if long_val_dataset is not None:
        # separately build a model for the longer sequence.
        # this is needed because the model doesn't support dynamic shapes.
        long_hparams_dict = dict(hparams_dict)
        long_hparams_dict['sequence_length'] = long_val_dataset.hparams.sequence_length
        # use smaller batch size for longer model to prevenet running out of memory
        long_hparams_dict['batch_size'] = model.hparams.batch_size // 2
        long_model = VideoPredictionModel(
            mode="test",  # to not build the losses and discriminators
            hparams_dict=long_hparams_dict,
            hparams=args.model_hparams,
            aggregate_nccl=args.aggregate_nccl)
        tf.get_variable_scope().reuse_variables()
        long_model.build_graph(long_val_dataset.make_batch(batch_size))
    else:
        long_model = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(train_dataset.hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    with tf.name_scope("parameter_count"):
        # exclude trainable variables that are replicas (used in multi-gpu setting)
        trainable_variables = set(tf.trainable_variables()) & set(model.saveable_variables)
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in trainable_variables])

    saver = tf.train.Saver(var_list=model.saveable_variables, max_to_keep=2)

    # None has the special meaning of evaluating at the end, so explicitly check for non-equality to zero
    if (args.summary_freq != 0 or args.image_summary_freq != 0 or
            args.eval_summary_freq != 0 or args.accum_eval_summary_freq != 0):
        summary_writer = tf.summary.FileWriter(args.output_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()
    max_steps = model.hparams.max_steps
    with tf.Session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.restore(sess, args.checkpoint)
        sess.run(model.post_init_ops)
        val_handle_eval = sess.run(val_handle)
        sess.graph.finalize()

        start_step = sess.run(global_step)

        def should(step, freq):
            if freq is None:
                return (step + 1) == (max_steps - start_step)
            else:
                return freq and ((step + 1) % freq == 0 or (step + 1) in (0, max_steps - start_step))

        def should_eval(step, freq):
            # never run eval summaries at the beginning since it's expensive, unless it's the last iteration
            return should(step, freq) and (step >= 0 or (step + 1) == (max_steps - start_step))

        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        for step in range(-1, max_steps - start_step):
            if step == 1:
                # skip step -1 and 0 for timing purposes (for warmstarting)
                start_time = time.time()

            fetches = {"global_step": global_step}
            if step >= 0:
                fetches["train_op"] = model.train_op
            if should(step, args.progress_freq):
                fetches['d_loss'] = model.d_loss
                fetches['g_loss'] = model.g_loss
                fetches['d_losses'] = model.d_losses
                fetches['g_losses'] = model.g_losses
                if isinstance(model.learning_rate, tf.Tensor):
                    fetches["learning_rate"] = model.learning_rate
            if should(step, args.summary_freq):
                fetches["summary"] = model.summary_op
            if should(step, args.image_summary_freq):
                fetches["image_summary"] = model.image_summary_op
            if should_eval(step, args.eval_summary_freq):
                fetches["eval_summary"] = model.eval_summary_op

            run_start_time = time.time()
            results = sess.run(fetches)
            run_elapsed_time = time.time() - run_start_time
            if run_elapsed_time > 1.5 and step > 0 and set(fetches.keys()) == {"global_step", "train_op"}:
                print('running train_op took too long (%0.1fs)' % run_elapsed_time)

            if (should(step, args.summary_freq) or
                    should(step, args.image_summary_freq) or
                    should_eval(step, args.eval_summary_freq)):
                val_fetches = {"global_step": global_step}
                if should(step, args.summary_freq):
                    val_fetches["summary"] = model.summary_op
                if should(step, args.image_summary_freq):
                    val_fetches["image_summary"] = model.image_summary_op
                if should_eval(step, args.eval_summary_freq):
                    val_fetches["eval_summary"] = model.eval_summary_op
                val_results = sess.run(val_fetches, feed_dict={train_handle: val_handle_eval})
                for name, summary in val_results.items():
                    if name == 'global_step':
                        continue
                    val_results[name] = add_tag_suffix(summary, '_1')

            if should(step, args.summary_freq):
                print("recording summary")
                summary_writer.add_summary(results["summary"], results["global_step"])
                summary_writer.add_summary(val_results["summary"], val_results["global_step"])
                print("done")
            if should(step, args.image_summary_freq):
                print("recording image summary")
                summary_writer.add_summary(results["image_summary"], results["global_step"])
                summary_writer.add_summary(val_results["image_summary"], val_results["global_step"])
                print("done")
            if should_eval(step, args.eval_summary_freq):
                print("recording eval summary")
                summary_writer.add_summary(results["eval_summary"], results["global_step"])
                summary_writer.add_summary(val_results["eval_summary"], val_results["global_step"])
                print("done")
            if should_eval(step, args.accum_eval_summary_freq):
                val_datasets = [val_dataset]
                val_models = [model]
                if long_model is not None:
                    val_datasets.append(long_val_dataset)
                    val_models.append(long_model)
                for i, (val_dataset_, val_model) in enumerate(zip(val_datasets, val_models)):
                    sess.run(val_model.accum_eval_metrics_reset_op)
                    # traverse (roughly up to rounding based on the batch size) all the validation dataset
                    accum_eval_summary_num_updates = val_dataset_.num_examples_per_epoch() // val_model.hparams.batch_size
                    val_fetches = {"global_step": global_step, "accum_eval_summary": val_model.accum_eval_summary_op}
                    for update_step in range(accum_eval_summary_num_updates):
                        print('evaluating %d / %d' % (update_step + 1, accum_eval_summary_num_updates))
                        val_results = sess.run(val_fetches, feed_dict={train_handle: val_handle_eval})
                    accum_eval_summary = add_tag_suffix(val_results["accum_eval_summary"], '_%d' % (i + 1))
                    print("recording accum eval summary")
                    summary_writer.add_summary(accum_eval_summary, val_results["global_step"])
                    print("done")
            if (should(step, args.summary_freq) or should(step, args.image_summary_freq) or
                    should_eval(step, args.eval_summary_freq) or should_eval(step, args.accum_eval_summary_freq)):
                summary_writer.flush()
            if should(step, args.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                # global step is read before it's incremented
                steps_per_epoch = train_dataset.num_examples_per_epoch() / batch_size
                train_epoch = results["global_step"] / steps_per_epoch
                print("progress  global step %d  epoch %0.1f" % (results["global_step"] + 1, train_epoch))
                if step > 0:
                    elapsed_time = time.time() - start_time
                    average_time = elapsed_time / step
                    images_per_sec = batch_size / average_time
                    remaining_time = (max_steps - (start_step + step + 1)) * average_time
                    print("          image/sec %0.1f  remaining %dm (%0.1fh) (%0.1fd)" %
                          (images_per_sec, remaining_time / 60, remaining_time / 60 / 60, remaining_time / 60 / 60 / 24))

                if results['d_losses']:
                    print("d_loss", results["d_loss"])
                for name, loss in results['d_losses'].items():
                    print("  ", name, loss)
                if results['g_losses']:
                    print("g_loss", results["g_loss"])
                for name, loss in results['g_losses'].items():
                    print("  ", name, loss)
                if isinstance(model.learning_rate, tf.Tensor):
                    print("learning_rate", results["learning_rate"])

            if should(step, args.save_freq):
                print("saving model to", args.output_dir)
                saver.save(sess, os.path.join(args.output_dir, "model"), global_step=global_step)
                print("done")


if __name__ == '__main__':
    main()
