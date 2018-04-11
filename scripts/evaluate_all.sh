# BAIR action-free robot pushing dataset
dataset=bair_action_free
for method_dir in \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    ours_vae_l2 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    sv2p_time_invariant \
; do
   CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair --dataset_hparams sequence_length=30 --checkpoint models/${dataset}/${method_dir} --mode test --results_dir results_test/${dataset} --batch_size 8
done

# KTH human actions dataset
# use batch_size=1 to ensure reproducibility when sampling subclips within a sequence
dataset=kth
for method_dir in \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    sv2p_time_variant \
    sv2p_time_invariant \
; do
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/kth --dataset_hparams sequence_length=40 --checkpoint models/${dataset}/${method_dir} --mode test --results_dir results_test/${dataset} --batch_size 1
done

# BAIR action-conditioned robot pushing dataset
dataset=bair
for method_dir in \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    ours_vae_l2 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    sna_l1 \
    sna_l2 \
    sv2p_time_variant \
; do
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --input_dir data/bair --dataset_hparams sequence_length=30 --checkpoint models/${dataset}/${method_dir} --mode test --results_dir results_test/${dataset} --batch_size 8
done
