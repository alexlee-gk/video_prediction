import argparse
import glob
import itertools
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset


class KTHVideoDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(KTHVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (64, 64, 3)

    def get_default_hparams_dict(self):
        default_hparams = super(KTHVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,
            sequence_length=20,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        return len(self.filenames)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def partition_data(input_dir):
    # List files and corresponding person IDs
    files = glob.glob(os.path.join(input_dir, '*/*.avi'))
    persons = np.array([int(f.split('/person')[1].split('_')[0]) for f in files])
    train_mask = persons <= 16

    train_fnames = [files[i] for i in np.where(train_mask)[0]]
    test_fnames = [files[i] for i in np.where(~train_mask)[0]]

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


def read_videos_and_save_tf_records(output_dir, fnames):
    def preprocess_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image[:, 20:-20], (64, 64), interpolation=cv2.INTER_LINEAR)
        return image.tostring()

    for i, fname in enumerate(fnames):
        output_fname = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + '.tfrecords')
        sequence = read_video(fname)
        save_tf_record(output_fname, [sequence], preprocess_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the directories "
                                                    "boxing, handclapping, handwaving, "
                                                    "jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    partition_names = ['train', 'val', 'test']
    partition_fnames = partition_data(args.input_dir)

    for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir, exist_ok=True)

        read_videos_and_save_tf_records(partition_dir, partition_fnames)


if __name__ == '__main__':
    main()
