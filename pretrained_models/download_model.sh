#!/usr/bin/env bash

# exit if any command fails
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DATASET_NAME MODEL_NAME" >&2
  exit 1
fi
DATASET_NAME=$1
MODEL_NAME=$2

declare -A model_name_to_fname
if [ ${DATASET_NAME} = "bair_action_free" ]; then
  model_name_to_fname=(
    [ours_deterministic]=${DATASET_NAME}_ours_deterministic_l1
    [ours_deterministic_l1]=${DATASET_NAME}_ours_deterministic_l1
    [ours_deterministic_l2]=${DATASET_NAME}_ours_deterministic_l2
    [ours_gan]=${DATASET_NAME}_ours_gan
    [ours_savp]=${DATASET_NAME}_ours_savp
    [ours_vae]=${DATASET_NAME}_ours_vae_l1
    [ours_vae_l1]=${DATASET_NAME}_ours_vae_l1
    [ours_vae_l2]=${DATASET_NAME}_ours_vae_l2
    [sv2p_time_invariant]=${DATASET_NAME}_sv2p_time_invariant
  )
elif [ ${DATASET_NAME} = "kth" ]; then
  model_name_to_fname=(
    [ours_deterministic]=${DATASET_NAME}_ours_deterministic_l1
    [ours_deterministic_l1]=${DATASET_NAME}_ours_deterministic_l1
    [ours_deterministic_l2]=${DATASET_NAME}_ours_deterministic_l2
    [ours_gan]=${DATASET_NAME}_ours_gan
    [ours_savp]=${DATASET_NAME}_ours_savp
    [ours_vae]=${DATASET_NAME}_ours_vae_l1
    [ours_vae_l1]=${DATASET_NAME}_ours_vae_l1
    [sv2p_time_invariant]=${DATASET_NAME}_sv2p_time_invariant
    [sv2p_time_variant]=${DATASET_NAME}_sv2p_time_variant
  )
elif [ ${DATASET_NAME} = "bair" ]; then
  model_name_to_fname=(
    [ours_deterministic]=${DATASET_NAME}_ours_deterministic_l1
    [ours_deterministic_l1]=${DATASET_NAME}_ours_deterministic_l1
    [ours_deterministic_l2]=${DATASET_NAME}_ours_deterministic_l2
    [ours_gan]=${DATASET_NAME}_ours_gan
    [ours_savp]=${DATASET_NAME}_ours_savp
    [ours_vae]=${DATASET_NAME}_ours_vae_l1
    [ours_vae_l1]=${DATASET_NAME}_ours_vae_l1
    [ours_vae_l2]=${DATASET_NAME}_ours_vae_l2
    [sna_l1]=${DATASET_NAME}_sna_l1
    [sna_l2]=${DATASET_NAME}_sna_l2
    [sv2p_time_variant]=${DATASET_NAME}_sv2p_time_variant
  )
else
  echo "Invalid dataset name: '${DATASET_NAME}' (choose from 'bair_action_free', 'kth', 'bair)" >&2
  exit 1
fi

if ! [[ ${model_name_to_fname[${MODEL_NAME}]} ]]; then
  echo "Invalid model name '${MODEL_NAME}' when dataset name is '${DATASET_NAME}'. Valid mode names are:" >&2
  for model_name in "${!model_name_to_fname[@]}"; do
    echo "'${model_name}'" >&2
  done
  exit 1
fi
TARGET_DIR=./pretrained_models/${DATASET_NAME}/${MODEL_NAME}
mkdir -p ${TARGET_DIR}
TAR_FNAME=${model_name_to_fname[${MODEL_NAME}]}.tar.gz
URL=http://rail.eecs.berkeley.edu/models/savp/pretrained_models/${TAR_FNAME}
echo "Downloading '${TAR_FNAME}'"
wget ${URL} -O ${TARGET_DIR}/${TAR_FNAME}
tar -xvf ${TARGET_DIR}/${TAR_FNAME} -C ${TARGET_DIR}
rm ${TARGET_DIR}/${TAR_FNAME}

echo "Succesfully finished downloading pretrained model '${MODEL_NAME}' on dataset '${DATASET_NAME}' into directory ${TARGET_DIR}"
