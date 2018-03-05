import argparse
import glob
import itertools
import os
import random
from multiprocessing import Pool

import cv2
import tensorflow as tf

from video_prediction.datasets.base_dataset import SequenceExampleVideoDataset


class UCF101VideoDataset(SequenceExampleVideoDataset):
    def __init__(self, *args, **kwargs):
        super(UCF101VideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'image/encoded', (240, 320, 3)

    def get_default_hparams_dict(self):
        default_hparams = super(UCF101VideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=4,
            sequence_length=12,
            random_crop_size=64,
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
        image_size = image_shape[-3:-1]
        random_crop_size = [self.hparams.random_crop_size] * 2
        crop_y = tf.random_uniform([], minval=0, maxval=image_size[0] - random_crop_size [0], dtype=tf.int32)
        crop_x = tf.random_uniform([], minval=0, maxval=image_size[1] - random_crop_size [1], dtype=tf.int32)
        crop_window = [crop_y, crop_x] + random_crop_size
        images = tf.map_fn(lambda image_buffer: tf.image.decode_and_crop_jpeg(image_buffer, crop_window),
                           image_buffers, dtype=tf.uint8, parallel_iterations=self.hparams.sequence_length)
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images.set_shape([None] + random_crop_size + [image_shape[-1]])
        # TODO: only random crop for training
        return images


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def partition_data(input_dir, train_test_list_dir):
    train_list_fnames = glob.glob(os.path.join(train_test_list_dir, 'trainlist*.txt'))
    test_list_fnames = glob.glob(os.path.join(train_test_list_dir, 'testlist*.txt'))

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

    train_fnames = [os.path.join(input_dir, train_fname) for train_fname in train_fnames]
    test_fnames = [os.path.join(input_dir, test_fname) for test_fname in test_fnames]

    random.shuffle(train_fnames)

    pivot = int(0.95 * len(train_fnames))
    train_fnames, val_fnames = train_fnames[:pivot], train_fnames[pivot:]
    return train_fnames, val_fnames, test_fnames


def read_video(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError
    vidcap = cv2.VideoCapture(fname)
    frames, (success, image) = [], vidcap.read()
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        success, image = vidcap.read()
    return frames


def save_tf_record(output_fname, sequences, preprocess_image):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            height, width, channels = sequence[0].shape
            context = tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
            })
            image_features = [_bytes_feature(preprocess_image(image)) for image in sequence]
            feature_lists = tf.train.FeatureLists(feature_list={
                'image/encoded': tf.train.FeatureList(feature=image_features),
            })
            example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
            writer.write(example.SerializeToString())


def read_videos_and_save_tf_records(output_dir, fnames, start_sequence_iter=None,
                                    end_sequence_iter=None, sequences_per_file=128):
    print('started process with PID:', os.getpid())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if start_sequence_iter is None:
        start_sequence_iter = 0
    if end_sequence_iter is None:
        end_sequence_iter = len(fnames)

    sess = tf.Session()
    image_ph = tf.placeholder(dtype=tf.uint8)
    image_jpeg = tf.image.encode_jpeg(image_ph, format='rgb', quality=100)
    encode_jpeg = lambda image: sess.run(image_jpeg, feed_dict={image_ph: image})

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
            save_tf_record(output_fname, sequences, encode_jpeg)
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

    partition_names = ['train', 'val', 'test']
    partition_fnames = partition_data(args.input_dir, args.train_test_list_dir)

    for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir, exist_ok=True)

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
