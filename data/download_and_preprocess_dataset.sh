#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 DATASET_NAME" >&2
  exit 1
fi
if [ $1 = "bair" ]; then
  TARGET_DIR=./data/bair
  mkdir -p ${TARGET_DIR}
  TAR_FNAME=bair_robot_pushing_dataset_v0.tar
  URL=http://rail.eecs.berkeley.edu/datasets/${TAR_FNAME}
  echo "Downloading $1 dataset (this takes a while)"
  wget ${URL} -O ${TARGET_DIR}/${TAR_FNAME}
  tar -xvf ${TARGET_DIR}/${TAR_FNAME} --strip-components=1 -C ${TARGET_DIR}
  rm ${TARGET_DIR}/${TAR_FNAME}
  mkdir -p ${TARGET_DIR}/val
  # reserve a fraction of the training set for validation
  mv ${TARGET_DIR}/train/traj_{256_to_511,512_to_767,768_to_1023,1024_to_1279,1280_to_1535,1536_to_1791,1792_to_2047,2048_to_2303,2304_to_2559}.tfrecords ${TARGET_DIR}/val/
elif [ $1 = "kth" ]; then
  TARGET_DIR=./data/kth
  mkdir -p ${TARGET_DIR}
  mkdir -p ${TARGET_DIR}/raw
  echo "Downloading $1 dataset (this takes a while)"
  for ACTION in walking jogging running boxing handwaving handclapping; do
    ZIP_FNAME=${ACTION}.zip
    URL=http://www.nada.kth.se/cvap/actions/${ZIP_FNAME}
    wget ${URL} -O ${TARGET_DIR}/raw/${ZIP_FNAME}
    unzip ${TARGET_DIR}/raw/${ZIP_FNAME} -d ${TARGET_DIR}/raw/${ACTION}
  done
  python video_prediction/datasets/kth_dataset.py ${TARGET_DIR}/raw ${TARGET_DIR}
  rm -rf ${TARGET_DIR}/raw
else
  echo "Invalid dataset name: '$1' (choose from 'bair', 'kth')" >&2
  exit 1
fi
echo "Succesfully finished downloading and preprocessing dataset $1"
