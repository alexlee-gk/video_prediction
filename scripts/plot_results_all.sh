python scripts/plot_results.py results_test/bair_action_free --method_dirs \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    ours_vae_l2 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    sv2p_time_invariant \
    svg_lp \
    --save --use_tex --plot_fname metrics_all.pdf

python scripts/plot_results.py results_test/bair --method_dirs \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    ours_vae_l2 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    sna_l1 \
    sna_l2 \
    sv2p_time_variant \
    --save --use_tex --plot_fname metrics_all.pdf

python scripts/plot_results.py results_test/kth --method_dirs \
    ours_vae_gan \
    ours_gan \
    ours_vae_l1 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    sv2p_time_variant \
    sv2p_time_invariant \
    svg_fp_resized_data_loader \
    --save --use_tex --plot_fname metrics_all.pdf


python scripts/plot_results.py results_test/bair_action_free --method_dirs \
    sv2p_time_invariant \
    svg_lp \
    ours_vae_gan \
    --save --use_tex --plot_fname metrics.pdf; \
python scripts/plot_results.py results_test/bair_action_free --method_dirs \
    ours_deterministic \
    ours_vae \
    ours_gan \
    ours_vae_gan \
    --save --use_tex --plot_fname metrics_ablation.pdf; \
python scripts/plot_results.py results_test/bair_action_free --method_dirs \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    ours_vae_l1 \
    ours_vae_l2 \
    --save --use_tex --plot_fname metrics_ablation_l1_l2.pdf; \
python scripts/plot_results.py results_test/kth --method_dirs \
    sv2p_time_variant \
    svg_fp_resized_data_loader \
    ours_vae_gan \
    --save --use_tex --plot_fname metrics.pdf; \
python scripts/plot_results.py results_test/kth --method_dirs \
    ours_deterministic \
    ours_vae \
    ours_gan \
    ours_vae_gan \
    --save --use_tex --plot_fname metrics_ablation.pdf; \
python scripts/plot_results.py results_test/bair --method_dirs \
    sv2p_time_variant \
    ours_deterministic \
    ours_vae \
    ours_gan \
    ours_vae_gan \
    --save --use_tex --plot_fname metrics.pdf; \
python scripts/plot_results.py results_test/bair --method_dirs \
    sna_l1 \
    sna_l2 \
    ours_deterministic_l1 \
    ours_deterministic_l2 \
    ours_vae_l1 \
    ours_vae_l2 \
    --save --use_tex --plot_fname metrics_ablation_l1_l2.pdf
