#!/usr/bin/env bash

# exit if any command fails
set -e

if [ "$#" -eq 2 ]; then
  if [ $1 = "bair" ]; then
    echo "IMAGE_SIZE argument is only applicable to kth dataset" >&2
    exit 1
  fi
elif [ "$#" -ne 1 ]; then
  echo "Usage: $0 DATASET_NAME [IMAGE_SIZE]" >&2
  exit 1
fi
if [ $1 = "bair" ]; then
  TARGET_DIR=./data/bair
  mkdir -p ${TARGET_DIR}
  TAR_FNAME=bair_robot_pushing_dataset_v0.tar
  URL=http://rail.eecs.berkeley.edu/datasets/${TAR_FNAME}
  echo "Downloading '$1' dataset (this takes a while)"
  wget ${URL} -O ${TARGET_DIR}/${TAR_FNAME}
  tar -xvf ${TARGET_DIR}/${TAR_FNAME} --strip-components=1 -C ${TARGET_DIR}
  rm ${TARGET_DIR}/${TAR_FNAME}
  mkdir -p ${TARGET_DIR}/val
  # reserve a fraction of the training set for validation
  mv ${TARGET_DIR}/train/traj_256_to_511.tfrecords ${TARGET_DIR}/val/
elif [ $1 = "kth" ]; then
  if [ "$#" -eq 2 ]; then
    IMAGE_SIZE=$2
    TARGET_DIR=./data/kth_${IMAGE_SIZE}
  else
    IMAGE_SIZE=64
    TARGET_DIR=./data/kth
  fi
  echo ${TARGET_DIR} ${IMAGE_SIZE}
  mkdir -p ${TARGET_DIR}
  mkdir -p ${TARGET_DIR}/raw
  echo "Downloading '$1' dataset (this takes a while)"
  for ACTION in walking jogging running boxing handwaving handclapping; do
    ZIP_FNAME=${ACTION}.zip
    URL=http://www.nada.kth.se/cvap/actions/${ZIP_FNAME}
    wget ${URL} -O ${TARGET_DIR}/raw/${ZIP_FNAME}
    unzip ${TARGET_DIR}/raw/${ZIP_FNAME} -d ${TARGET_DIR}/raw/${ACTION}
  done
  FRAME_RATE=25
  mkdir -p ${TARGET_DIR}/processed
  # download files with metadata specifying the subsequences
  TAR_FNAME=kth_meta.tar.gz
  URL=http://rail.eecs.berkeley.edu/models/savp/data/${TAR_FNAME}
  echo "Downloading '${TAR_FNAME}'"
  wget ${URL} -O ${TARGET_DIR}/processed/${TAR_FNAME}
  tar -xzvf ${TARGET_DIR}/processed/${TAR_FNAME} --strip 1 -C ${TARGET_DIR}/processed
  # convert the videos into sequence of downscaled images
  echo "Processing '$1' dataset"
  for ACTION in walking jogging running boxing handwaving handclapping; do
    for VIDEO_FNAME in ${TARGET_DIR}/raw/${ACTION}/*.avi; do
      FNAME=$(basename ${VIDEO_FNAME})
      FNAME=${FNAME%_uncomp.avi}
      # sometimes the directory is not created, so try until it is
      while [ ! -d "${TARGET_DIR}/processed/${ACTION}/${FNAME}" ]; do
        mkdir -p ${TARGET_DIR}/processed/${ACTION}/${FNAME}
      done
      ffmpeg -i ${VIDEO_FNAME} -r ${FRAME_RATE} -f image2 -s ${IMAGE_SIZE}x${IMAGE_SIZE} \
      ${TARGET_DIR}/processed/${ACTION}/${FNAME}/image-%03d_${IMAGE_SIZE}x${IMAGE_SIZE}.png
    done
  done
  python video_prediction/datasets/kth_dataset.py ${TARGET_DIR}/processed ${TARGET_DIR} ${IMAGE_SIZE}
  rm -rf ${TARGET_DIR}/raw
  rm -rf ${TARGET_DIR}/processed
else
  echo "Invalid dataset name: '$1' (choose from 'bair', 'kth')" >&2
  exit 1
fi
echo "Succesfully finished downloading and preprocessing dataset '$1'"
