# BAIR action-free robot pushing dataset
dataset=bair_action_free
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair --dataset bair \
    --dataset_hparams sequence_length=30 --model ground_truth --mode test \
    --output_gif_dir results_test_2afc/${dataset}/ground_truth \
    --output_png_dir results_test_samples/${dataset}/ground_truth --gif_length 10
for method_dir in \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    sv2p_time_invariant \
; do
    CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair \
        --dataset_hparams sequence_length=30 --checkpoint models/${dataset}/${method_dir} --mode test \
        --results_gif_dir results_test_2afc/${dataset} \
        --results_png_dir results_test_samples/${dataset} --gif_length 10
done

# KTH human actions dataset
# use batch_size=1 to ensure reproducibility when sampling subclips within a sequence
dataset=kth
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/kth --dataset kth \
    --dataset_hparams sequence_length=40 --model ground_truth --mode test \
    --output_gif_dir results_test_2afc/${dataset}/ground_truth \
    --output_png_dir results_test_samples/${dataset}/ground_truth --gif_length 10 --batch_size 1
for method_dir in \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    sv2p_time_invariant \
    sv2p_time_variant \
; do
    CUDA_VISIBLE_DEVICES=1 python scripts/generate.py --input_dir data/kth \
        --dataset_hparams sequence_length=40 --checkpoint models/${dataset}/${method_dir} --mode test \
        --results_gif_dir results_test_2afc/${dataset} \
        --results_png_dir results_test_samples/${dataset} --gif_length 10 --batch_size 1
done

# BAIR action-conditioned robot pushing dataset
dataset=bair
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair --dataset bair \
    --dataset_hparams sequence_length=30 --model ground_truth --mode test \
    --output_gif_dir results_test_2afc/${dataset}/ground_truth \
    --output_png_dir results_test_samples/${dataset}/ground_truth --gif_length 10
for method_dir in \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    sv2p_time_variant \
; do
    CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair \
        --dataset_hparams sequence_length=30 --checkpoint models/${dataset}/${method_dir} --mode test \
        --results_gif_dir results_test_2afc/${dataset} \
        --results_png_dir results_test_samples/${dataset} --gif_length 10
done
