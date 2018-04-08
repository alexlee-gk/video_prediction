# BAIR action-free robot pushing dataset
for model in \
  ours_deterministic_l1 \
  ours_deterministic_l2 \
  ours_vae_l1 \
  ours_vae_l2 \
  ours_gan \
  ours_savp \
; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/bair --dataset bair --model savp --model_hparams_dict hparams/bair_action_free/${model}/model_hparams.json --output_dir logs/bair_action_free/${model}
done

# KTH human actions dataset
for model in \
  ours_deterministic_l1 \
  ours_deterministic_l2 \
  ours_vae_l1 \
  ours_gan \
  ours_savp \
; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/kth --dataset kth --model savp --model_hparams_dict hparams/kth/${model}/model_hparams.json --output_dir logs/kth/${model}
done

# BAIR action-conditioned robot pushing dataset
for model in \
  ours_deterministic_l1 \
  ours_deterministic_l2 \
  ours_vae_l1 \
  ours_vae_l2 \
  ours_gan \
  ours_savp \
; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/bair --dataset bair --dataset_hparams use_state=True --model savp --model_hparams_dict hparams/bair/${model}/model_hparams.json --output_dir logs/bair/${model}
done
for model in \
  sna_l1 \
  sna_l2 \
; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/bair --dataset bair --dataset_hparams use_state=True --model sna --model_hparams_dict hparams/bair/${model}/model_hparams.json --output_dir logs/bair/${model}
done
