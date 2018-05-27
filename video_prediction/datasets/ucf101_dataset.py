import argparse
import glob
import itertools
import os
import random
import re
from multiprocessing import Pool

import cv2
import tensorflow as tf

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset


class UCF101VideoDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(UCF101VideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (240, 320, 3)

    def get_default_hparams_dict(self):
        default_hparams = super(UCF101VideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=4,
            sequence_length=8,
            random_crop_size=0,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return True

    def decode_and_preprocess_images(self, image_buffers, image_shape):
        if self.hparams.crop_size:
            raise NotImplementedError
        if self.hparams.scale_size:
            raise NotImplementedError
        image_buffers = tf.reshape(image_buffers, [-1])
        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)
        image_size = tf.image.extract_jpeg_shape(image_buffers[0])[:2]  # should be the same as image_shape[:2]
        if self.hparams.random_crop_size:
            random_crop_size = [self.hparams.random_crop_size] * 2
            crop_y = tf.random_uniform([], minval=0, maxval=image_size[0] - random_crop_size[0], dtype=tf.int32)
            crop_x = tf.random_uniform([], minval=0, maxval=image_size[1] - random_crop_size[1], dtype=tf.int32)
            crop_window = [crop_y, crop_x] + random_crop_size
            images = [tf.image.decode_and_crop_jpeg(image_buffer, crop_window) for image_buffer in image_buffers]
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
            images.set_shape([None] + random_crop_size + [image_shape[-1]])
        else:
            images = [tf.image.decode_jpeg(image_buffer) for image_buffer in image_buffers]
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
            images.set_shape([None] + list(image_shape))
        # TODO: only random crop for training
        return images

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('sequence_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def partition_data(input_dir, train_test_list_dir):
    train_list_fnames = glob.glob(os.path.join(train_test_list_dir, 'trainlist*.txt'))
    test_list_fnames = glob.glob(os.path.join(train_test_list_dir, 'testlist*.txt'))
    test_list_fnames_mathieu = [os.path.join(train_test_list_dir, 'testlist01.txt')]

    def read_fnames(list_fnames):
        fnames = []
        for list_fname in sorted(list_fnames):
            with open(list_fname, 'r') as f:
                while True:
                    fname = f.readline()
                    if not fname:
                        break
                    fnames.append(fname.split('\n')[0].split(' ')[0])
        return fnames

    train_fnames = read_fnames(train_list_fnames)
    test_fnames = read_fnames(test_list_fnames)
    test_fnames_mathieu = read_fnames(test_list_fnames_mathieu)

    train_fnames = [os.path.join(input_dir, train_fname) for train_fname in train_fnames]
    test_fnames = [os.path.join(input_dir, test_fname) for test_fname in test_fnames]
    test_fnames_mathieu = [os.path.join(input_dir, test_fname) for test_fname in test_fnames_mathieu]
    # only use every 10 videos as in Mathieu et al.
    test_fnames_mathieu = test_fnames_mathieu[::10]

    random.shuffle(train_fnames)

    pivot = int(0.95 * len(train_fnames))
    train_fnames, val_fnames = train_fnames[:pivot], train_fnames[pivot:]
    return train_fnames, val_fnames, test_fnames, test_fnames_mathieu


def read_video(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError
    vidcap = cv2.VideoCapture(fname)
    frames, (success, image) = [], vidcap.read()
    while success:
        frames.append(image)
        success, image = vidcap.read()
    return frames


def save_tf_record(output_fname, sequences, preprocess_image):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            encoded_sequence = [preprocess_image(image) for image in sequence]
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _bytes_list_feature(encoded_sequence),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def read_videos_and_save_tf_records(output_dir, fnames, start_sequence_iter=None,
                                    end_sequence_iter=None, sequences_per_file=128):
    print('started process with PID:', os.getpid())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if start_sequence_iter is None:
        start_sequence_iter = 0
    if end_sequence_iter is None:
        end_sequence_iter = len(fnames)

    def preprocess_image(image):
        if image.shape != (240, 320, 3):
            image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)
        return tf.compat.as_bytes(cv2.imencode(".jpg", image)[1].tobytes())

    print('reading and saving sequences {0} to {1}'.format(start_sequence_iter, end_sequence_iter))

    sequences = []
    for sequence_iter in range(start_sequence_iter, end_sequence_iter):
        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)

        sequences.append(read_video(fnames[sequence_iter]))

        if len(sequences) == sequences_per_file or sequence_iter == (end_sequence_iter - 1):
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter)
            output_fname = os.path.join(output_dir, output_fname)
            save_tf_record(output_fname, sequences, preprocess_image)
            sequences[:] = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the directories of "
                                                    "classes, each of which contains avi files.")
    parser.add_argument("train_test_list_dir", type=str, help='directory containing trainlist*.txt'
                                                              'and testlist*.txt files.')
    parser.add_argument("output_dir", type=str)
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel workers')
    args = parser.parse_args()

    partition_names = ['train', 'val', 'test', 'test_mathieu']
    partition_fnames = partition_data(args.input_dir, args.train_test_list_dir)

    for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)

        if args.num_workers > 1:
            num_seqs_per_worker = len(partition_fnames) // args.num_workers
            start_seq_iters = [num_seqs_per_worker * i for i in range(args.num_workers)]
            end_seq_iters = [num_seqs_per_worker * (i + 1) - 1 for i in range(args.num_workers)]
            end_seq_iters[-1] = len(partition_fnames)

            p = Pool(args.num_workers)
            p.starmap(read_videos_and_save_tf_records, zip([partition_dir] * args.num_workers,
                                                           [partition_fnames] * args.num_workers,
                                                           start_seq_iters, end_seq_iters))
        else:
            read_videos_and_save_tf_records(partition_dir, partition_fnames)


if __name__ == '__main__':
    main()
