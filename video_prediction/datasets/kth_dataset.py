import argparse
import glob
import itertools
import os
import pickle
import random
import re

import numpy as np
import skimage.io
import tensorflow as tf

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset


class KTHVideoDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(KTHVideoDataset, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['height', 'width', 'channels'])
        self.state_like_names_and_shapes['images'] = 'images/encoded', image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(KTHVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,
            sequence_length=20,
            long_sequence_length=40,
            force_time_shift=True,
            shuffle_on_val=True,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        with open(os.path.join(self.input_dir, 'sequence_lengths.txt'), 'r') as sequence_lengths_file:
            sequence_lengths = sequence_lengths_file.readlines()
        sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
        return np.sum(np.array(sequence_lengths) >= self.hparams.sequence_length)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def partition_data(input_dir):
    # List files and corresponding person IDs
    fnames = glob.glob(os.path.join(input_dir, '*/*'))
    fnames = [fname for fname in fnames if os.path.isdir(fname)]

    persons = [re.match('person(\d+)_\w+_\w+', os.path.split(fname)[1]).group(1) for fname in fnames]
    persons = np.array([int(person) for person in persons])

    train_mask = persons <= 16

    train_fnames = [fnames[i] for i in np.where(train_mask)[0]]
    test_fnames = [fnames[i] for i in np.where(~train_mask)[0]]

    random.shuffle(train_fnames)

    pivot = int(0.95 * len(train_fnames))
    train_fnames, val_fnames = train_fnames[:pivot], train_fnames[pivot:]
    return train_fnames, val_fnames, test_fnames


def save_tf_record(output_fname, sequences):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            encoded_sequence = [image.tostring() for image in sequence]
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _bytes_list_feature(encoded_sequence),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def read_frames_and_save_tf_records(output_dir, video_dirs, image_size, sequences_per_file=128):
    partition_name = os.path.split(output_dir)[1]

    sequences = []
    sequence_iter = 0
    sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    for video_iter, video_dir in enumerate(video_dirs):
        meta_partition_name = partition_name if partition_name == 'test' else 'train'
        meta_fname = os.path.join(os.path.split(video_dir)[0], '%s_meta%dx%d.pkl' %
                                  (meta_partition_name, image_size, image_size))
        with open(meta_fname, "rb") as f:
            data = pickle.load(f)

        vid = os.path.split(video_dir)[1]
        (d,) = [d for d in data if d['vid'] == vid]
        for frame_fnames_iter, frame_fnames in enumerate(d['files']):
            frame_fnames = [os.path.join(video_dir, frame_fname) for frame_fname in frame_fnames]
            frames = skimage.io.imread_collection(frame_fnames)
            # they are grayscale images, so just keep one of the channels
            frames = [frame[..., 0:1] for frame in frames]

            if not sequences:
                last_start_sequence_iter = sequence_iter
                print("reading sequences starting at sequence %d" % sequence_iter)

            sequences.append(frames)
            sequence_iter += 1
            sequence_lengths_file.write("%d\n" % len(frames))

            if (len(sequences) == sequences_per_file or
                    (video_iter == (len(video_dirs) - 1) and frame_fnames_iter == (len(d['files']) - 1))):
                output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
                output_fname = os.path.join(output_dir, output_fname)
                save_tf_record(output_fname, sequences)
                sequences[:] = []
    sequence_lengths_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories "
                                                    "boxing, handclapping, handwaving, "
                                                    "jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("image_size", type=int)
    args = parser.parse_args()

    partition_names = ['train', 'val', 'test']
    partition_fnames = partition_data(args.input_dir)

    for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        read_frames_and_save_tf_records(partition_dir, partition_fnames, args.image_size)


if __name__ == '__main__':
    main()
