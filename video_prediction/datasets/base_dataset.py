import glob
import os
import random
import re
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams


class BaseVideoDataset(object):
    def __init__(self, input_dir, mode='train', num_epochs=None, seed=None,
                 hparams_dict=None, hparams=None):
        """
        Args:
            input_dir: either a directory containing subdirectories train,
                val, test, etc, or a directory containing the tfrecords.
            mode: either train, val, or test
            num_epochs: if None, dataset is iterated indefinitely.
            seed: random seed for the op that samples subsequences.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).

        Note:
            self.input_dir is the directory containing the tfrecords.
        """
        self.input_dir = os.path.normpath(os.path.expanduser(input_dir))
        self.mode = mode
        self.num_epochs = num_epochs
        self.seed = seed

        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.filenames = None
        # look for tfrecords in input_dir and input_dir/mode directories
        for input_dir in [self.input_dir, os.path.join(self.input_dir, self.mode)]:
            filenames = glob.glob(os.path.join(input_dir, '*.tfrecord*'))
            if filenames:
                self.input_dir = input_dir
                self.filenames = sorted(filenames)  # ensures order is the same across systems
                break
        if not self.filenames:
            raise FileNotFoundError('No tfrecords were found in %s.' % self.input_dir)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])

        self.state_like_names_and_shapes = OrderedDict()
        self.action_like_names_and_shapes = OrderedDict()

        self.hparams = self.parse_hparams(hparams_dict, hparams)

    def get_default_hparams_dict(self):
        """
        Returns:
            A dict with the following hyperparameters.

            crop_size: crop image into a square with sides of this length.
            scale_size: resize image to this size after it has been cropped.
            context_frames: the number of ground-truth frames to pass in at
                start.
            sequence_length: the number of frames in the video sequence, so
                state-like sequences are of length sequence_length and
                action-like sequences are of length sequence_length - 1.
                This number includes the context frames.
            frame_skip: number of frames to skip in between outputted frames,
                so frame_skip=0 denotes no skipping.
            time_shift: shift in time by multiples of this, so time_shift=1
                denotes all possible shifts. time_shift=0 denotes no shifting.
                It is ignored (equiv. to time_shift=0) when mode != 'train'.
            force_time_shift: whether to do the shift in time regardless of
                mode.
            use_state: whether to load and return state and actions.
        """
        hparams = dict(
            crop_size=0,
            scale_size=0,
            context_frames=1,
            sequence_length=0,
            frame_skip=0,
            time_shift=1,
            force_time_shift=False,
            use_state=False,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self, hparams_dict, hparams):
        parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        return parsed_hparams

    @property
    def jpeg_encoding(self):
        raise NotImplementedError

    def set_sequence_length(self, sequence_length):
        self.hparams.sequence_length = sequence_length

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example or tf.train.SequenceExample into
        images, states, actions, etc tensors.
        """
        raise NotImplementedError

    def make_batch(self, batch_size):
        filenames = self.filenames
        if self.mode == 'train':
            random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parser, num_parallel_calls=batch_size)
        dataset.prefetch(2 * batch_size)

        # if self.mode == 'train':
        #     min_queue_examples = int(
        #         self.num_examples_per_epoch() * 0.4)
        #     # Ensure that the capacity is sufficiently large to provide good random
        #     # shuffling.
        #     dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=128 * 8)

        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        state_like_batches, action_like_batches = iterator.get_next()

        input_batches = OrderedDict(list(state_like_batches.items()) + list(action_like_batches.items()))
        for input_batch in input_batches.values():
            input_batch.set_shape([batch_size] + [None] * (input_batch.shape.ndims - 1))
        target_batches = state_like_batches['images'][:, self.hparams.context_frames:]
        return input_batches, target_batches

    def decode_and_preprocess_images(self, image_buffers, image_shape):
        def decode_and_preprocess_image(image_buffer):
            image_buffer = tf.reshape(image_buffer, [])
            if self.jpeg_encoding:
                image = tf.image.decode_jpeg(image_buffer)
            else:
                image = tf.decode_raw(image_buffer, tf.uint8)
            image = tf.reshape(image, image_shape)
            crop_size = self.hparams.crop_size
            scale_size = self.hparams.scale_size
            if crop_size or scale_size:
                if not crop_size:
                    crop_size = min(image_shape[0], image_shape[1])
                image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
                image = tf.reshape(image, [crop_size, crop_size, 3])
                if scale_size:
                    # upsample with bilinear interpolation but downsample with area interpolation
                    if crop_size < scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.BILINEAR)
                    elif crop_size > scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.AREA)
                    else:
                        # image remains unchanged
                        pass
            return image

        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)
        images = [decode_and_preprocess_image(image_buffer) for image_buffer in image_buffers]
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        return images

    def slice_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        """
        Slices sequences of length `example_sequence_length` into subsequences
        of length `sequence_length`. The dicts of sequences are updated
        in-place and the same dicts are returned.
        """
        # handle random shifting and frame skip
        sequence_length = self.hparams.sequence_length  # desired sequence length
        frame_skip = self.hparams.frame_skip
        time_shift = self.hparams.time_shift
        if isinstance(example_sequence_length, tf.Tensor):
            example_sequence_length = tf.cast(example_sequence_length, tf.int32)
        if (time_shift > 0 and self.mode == 'train') or self.hparams.force_time_shift:
            assert time_shift > 0 and isinstance(time_shift, int)
            num_shifts = ((example_sequence_length - 1) - (sequence_length - 1) * (frame_skip + 1)) // time_shift
            assert_message = ('example_sequence_length has to be at least %d when '
                              'sequence_length=%d, frame_skip=%d.' %
                              ((sequence_length - 1) * (frame_skip + 1) + 1,
                               sequence_length, frame_skip))
            with tf.control_dependencies([tf.assert_greater_equal(num_shifts, 0,
                    data=[example_sequence_length, num_shifts], message=assert_message)]):
                t_start = tf.random_uniform([], 0, num_shifts + 1, dtype=tf.int32, seed=self.seed) * time_shift
        elif time_shift < 0:  # if negative, always use the last subsequence
            t_start = ((example_sequence_length - 1) - (sequence_length - 1) * (frame_skip + 1))
        else:
            t_start = 0
        state_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1) + 1, frame_skip + 1)
        action_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1))

        for example_name, seq in state_like_seqs.items():
            seq = tf.convert_to_tensor(seq)[state_like_t_slice]
            seq.set_shape([sequence_length] + seq.shape.as_list()[1:])
            state_like_seqs[example_name] = seq
        for example_name, seq in action_like_seqs.items():
            seq = tf.convert_to_tensor(seq)[action_like_t_slice]
            seq.set_shape([(sequence_length - 1) * (frame_skip + 1)] + seq.shape.as_list()[1:])
            # concatenate actions of skipped frames into single macro actions
            seq = tf.reshape(seq, [sequence_length - 1, -1])
            action_like_seqs[example_name] = seq
        return state_like_seqs, action_like_seqs

    def num_examples_per_epoch(self):
        raise NotImplementedError


class VideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where a sequence is stored as
    multiple tf.train.Example and each of them is stored under a different
    feature name (which is indexed by the time step).
    """
    def __init__(self, *args, **kwargs):
        super(VideoDataset, self).__init__(*args, **kwargs)
        self._max_sequence_length = None
        self._dict_message = None

    def _check_or_infer_shapes(self):
        """
        Should be called after state_like_names_and_shapes and
        action_like_names_and_shapes have been finalized.
        """
        state_like_names_and_shapes = OrderedDict([(k, list(v)) for k, v in self.state_like_names_and_shapes.items()])
        action_like_names_and_shapes = OrderedDict([(k, list(v)) for k, v in self.action_like_names_and_shapes.items()])
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        self._dict_message = MessageToDict(tf.train.Example.FromString(example))
        for example_name, name_and_shape in (list(state_like_names_and_shapes.items()) +
                                             list(action_like_names_and_shapes.items())):
            name, shape = name_and_shape
            feature = self._dict_message['features']['feature']
            names = [name_ for name_ in feature.keys() if re.search(name.replace('%d', '\d+'), name_) is not None]
            if not names:
                raise ValueError('Could not found any feature with name pattern %s.' % name)
            if example_name in self.state_like_names_and_shapes:
                sequence_length = len(names)
            else:
                sequence_length = len(names) + 1
            if self._max_sequence_length is None:
                self._max_sequence_length = sequence_length
            else:
                self._max_sequence_length = min(sequence_length, self._max_sequence_length)
            name = names[0]
            feature = feature[name]
            list_type, = feature.keys()
            if list_type == 'floatList':
                inferred_shape = (len(feature[list_type]['value']),)
                if shape is None:
                    name_and_shape[1] = inferred_shape
                else:
                    if inferred_shape != shape:
                        raise ValueError('Inferred shape for feature %s is %r but instead got shape %r.' %
                                         (name, inferred_shape, shape))
            elif list_type == 'bytesList':
                image_str, = feature[list_type]['value']
                # try to infer image shape
                inferred_shape = None
                if not self.jpeg_encoding:
                    spatial_size = len(image_str) // 4
                    height = width = int(np.sqrt(spatial_size))  # assume square image
                    if len(image_str) == (height * width * 4):
                        inferred_shape = (height, width, 3)
                if shape is None:
                    if inferred_shape is not None:
                        name_and_shape[1] = inferred_shape
                    else:
                        raise ValueError('Unable to infer shape for feature %s of size %d.' % (name, len(image_str)))
                else:
                    if inferred_shape is not None and inferred_shape != shape:
                        raise ValueError('Inferred shape for feature %s is %r but instead got shape %r.' %
                                         (name, inferred_shape, shape))
            else:
                raise NotImplementedError
        self.state_like_names_and_shapes = OrderedDict([(k, tuple(v)) for k, v in state_like_names_and_shapes.items()])
        self.action_like_names_and_shapes = OrderedDict([(k, tuple(v)) for k, v in action_like_names_and_shapes.items()])

        # set sequence_length to the longest possible if it is not specified
        if not self.hparams.sequence_length:
            self.hparams.sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1

    def set_sequence_length(self, sequence_length):
        if not sequence_length:
            sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1
        self.hparams.sequence_length = sequence_length

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into images, states, actions, etc tensors.
        """
        features = dict()
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                if example_name.startswith('images'):  # special handling for image
                    features[name % i] = tf.FixedLenFeature([1], tf.string)
                else:
                    features[name % i] = tf.FixedLenFeature(shape, tf.float32)
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                features[name % i] = tf.FixedLenFeature(shape, tf.float32)

        # check that the features are in the tfrecord
        for name in features.keys():
            if name not in self._dict_message['features']['feature']:
                raise ValueError('Feature with name %s not found in tfrecord. Possible feature names are:\n%s' %
                                 (name, '\n'.join(sorted(self._dict_message['features']['feature'].keys()))))

        # parse all the features of all time steps together
        features = tf.parse_single_example(serialized_example, features=features)

        state_like_seqs = OrderedDict([(example_name, []) for example_name in self.state_like_names_and_shapes])
        action_like_seqs = OrderedDict([(example_name, []) for example_name in self.action_like_names_and_shapes])
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                state_like_seqs[example_name].append(features[name % i])
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                action_like_seqs[example_name].append(features[name % i])

        # for this class, it's much faster to decode and preprocess the entire sequence before sampling a slice
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name.startswith('images'):
                state_like_seqs[example_name] = self.decode_and_preprocess_images(state_like_seqs[example_name], shape)

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, self._max_sequence_length)
        return state_like_seqs, action_like_seqs


class SequenceExampleVideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where an entire sequence is stored as
    a single tf.train.SequenceExample.
    """
    def parser(self, serialized_example):
        """
        Parses a single tf.train.SequenceExample into images, states, actions, etc tensors.
        """
        sequence_features = dict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name.startswith('images'):  # special handling for image
                sequence_features[name] = tf.FixedLenSequenceFeature([1], tf.string)
            else:
                sequence_features[name] = tf.FixedLenSequenceFeature(shape, tf.float32)
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            sequence_features[name] = tf.FixedLenSequenceFeature(shape, tf.float32)

        _, sequence_features = tf.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

        state_like_seqs = OrderedDict()
        action_like_seqs = OrderedDict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            state_like_seqs[example_name] = sequence_features[name]
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            action_like_seqs[example_name] = sequence_features[name]

        # the sequence_length of this example is determined by the shortest sequence
        example_sequence_length = []
        for example_name, seq in state_like_seqs.items():
            example_sequence_length.append(tf.shape(seq)[0])
        for example_name, seq in action_like_seqs.items():
            example_sequence_length.append(tf.shape(seq)[0] + 1)
        example_sequence_length = tf.reduce_min(example_sequence_length)

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        # decode and preprocess images on the sampled slice only
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name.startswith('images'):
                state_like_seqs[example_name] = self.decode_and_preprocess_images(state_like_seqs[example_name], shape)
        return state_like_seqs, action_like_seqs


class VarLenFeatureVideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where an entire sequence is stored as
    a single tf.train.Example.

    https://github.com/tensorflow/tensorflow/issues/15977
    """
    def parser(self, serialized_example):
        """
        Parses a single tf.train.SequenceExample into images, states, actions, etc tensors.
        """
        features = dict()
        features['sequence_length'] = tf.FixedLenFeature((), tf.int64)
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name.startswith('images'):
                features[name] = tf.VarLenFeature(tf.string)
            else:
                features[name] = tf.VarLenFeature(tf.float32)
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            features[name] = tf.VarLenFeature(tf.float32)

        features = tf.parse_single_example(serialized_example, features=features)

        example_sequence_length = features['sequence_length']

        state_like_seqs = OrderedDict()
        action_like_seqs = OrderedDict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name.startswith('images'):
                seq = tf.sparse_tensor_to_dense(features[name], '')
            else:
                seq = tf.sparse_tensor_to_dense(features[name])
                seq = tf.reshape(seq, [example_sequence_length] + list(shape))
            state_like_seqs[example_name] = seq
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            seq = tf.sparse_tensor_to_dense(features[name])
            seq = tf.reshape(seq, [example_sequence_length - 1] + list(shape))
            action_like_seqs[example_name] = seq

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        # decode and preprocess images on the sampled slice only
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name.startswith('images'):
                state_like_seqs[example_name] = self.decode_and_preprocess_images(state_like_seqs[example_name], shape)
        return state_like_seqs, action_like_seqs


if __name__ == '__main__':
    import cv2
    from video_prediction import datasets

    datasets = [
        datasets.GoogleRobotVideoDataset('data/push/push_testseen', mode='test'),
        datasets.SV2PVideoDataset('data/shape', mode='val'),
        datasets.SV2PVideoDataset('data/humans', mode='val'),
        datasets.SoftmotionVideoDataset('data/softmotion30_v1', mode='val'),
        datasets.KTHVideoDataset('data/kth', mode='val'),
        datasets.UCF101VideoDataset('data/ucf101', mode='val'),
    ]
    batch_size = 4

    sess = tf.Session()

    for dataset in datasets:
        inputs, _ = dataset.make_batch(batch_size)
        images = inputs['images']
        images = tf.reshape(images, [-1] + images.get_shape().as_list()[2:])
        images = sess.run(images)
        images = (images * 255).astype(np.uint8)
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(dataset.input_dir, image)
            cv2.waitKey(50)
