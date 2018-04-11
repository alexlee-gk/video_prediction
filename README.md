# Stochastic Adversarial Video Prediction
[[Project Page]](https://alexlee-gk.github.io/video_prediction/) [[Paper]](https://arxiv.org/abs/1804.01523)

TensorFlow implementation for stochastic adversarial video prediction. Given a sequence of initial frames, our model is able to predict future frames of various possible futures. For example, in the next two sequences, we show the ground truth sequence on the left and random predictions of our model on the right. Predicted frames are indicated by the yellow bar at the bottom. For more examples, visit the [project page](https://alexlee-gk.github.io/video_prediction/).

<img src="https://alexlee-gk.github.io/video_prediction/index_files/images/bair_action_free_random_00066_crop.gif" height="96">
<img src="https://alexlee-gk.github.io/video_prediction/index_files/images/bair_action_free_random_00006_crop.gif" height="96">

**Stochastic Adversarial Video Prediction,**  
[Alex X. Lee](https://people.eecs.berkeley.edu/~alexlee_gk/), [Richard Zhang](https://richzhang.github.io/), [Frederik Ebert](https://febert.github.io/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Chelsea Finn](https://people.eecs.berkeley.edu/~cbfinn/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).  
arXiv preprint arXiv:1804.01523, 2018.

## Getting Started ###
### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation
- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/alexlee-gk/video_prediction.git
cd video_prediction
```
- Install TensorFlow >= 1.5 and dependencies from http://tensorflow.org/
- Install ffmpeg (optional, used to generate GIFs for visualization, e.g. in TensorBoard)
- Install other dependencies
```bash
pip install -r requirements.txt
```

### Use a Pre-trained Model
- Download and preprocess a dataset (e.g. `bair`):
```bash
bash data/download_and_preprocess_dataset.sh bair
```
- Download a pre-trained model (e.g. `ours_savp`) for the action-free version of that dataset (i.e. `bair_action_free`):
```bash
bash pretrained_models/download_model.sh bair_action_free ours_savp
```
- Sample predictions from the model:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair \
  --dataset_hparams sequence_length=30 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --results_dir results_test_samples/bair_action_free
```
- The predictions are saved as images and GIFs in `results_test_samples/bair_action_free/ours_savp`.
- Evaluate predictions from the model using full-reference metrics:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair \
  --dataset_hparams sequence_length=30 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --results_dir results_test/bair_action_free
```
- The results are saved in `results_test/bair_action_free/ours_savp`.
- See evaluation details of our experiments in [`scripts/generate_all.sh`](scripts/generate_all.sh) and [`scripts/evaluate_all.sh`](scripts/evaluate_all.sh).

### Model Training
- To train a model, download and preprocess a dataset (e.g. `bair`):
```bash
bash data/download_and_preprocess_dataset.sh bair
```
- Train a model (e.g. our SAVP model on the BAIR action-free robot pushing dataset):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/bair --dataset bair \
  --model savp --model_hparams_dict hparams/bair_action_free/ours_savp/model_hparams.json \
  --output_dir logs/bair_action_free/ours_svap
```
- To view training and validation information (e.g. loss plots, GIFs of predictions), run `tensorboard --logdir logs/bair_action_free --port 6006` and open http://localhost:6006.
- For multi-GPU training, set `CUDA_VISIBLE_DEVICES` to a comma-separated list of devices, e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`. To use the CPU, set `CUDA_VISIBLE_DEVICES=""`.
- See more training details for other datasets and models in [`scripts/train_all.sh`](scripts/train_all.sh).

### Datasets
Download the datasets using the following script. These datasets are collected by other researchers. Please cite their papers if you use the data.
- Download and preprocess the dataset.
```bash
bash data/download_and_preprocess_dataset.sh dataset_name
```
The `dataset_name` should be one of the following:
- `bair`: [BAIR robot pushing dataset](https://sites.google.com/view/sna-visual-mpc/). [[Citation](data/bibtex/sna.txt)]
- `kth`: [KTH human actions dataset](http://www.nada.kth.se/cvap/actions/). [[Citation](data/bibtex/kth.txt)]

To use a different dataset, preprocess it into TFRecords files and define a class for it. See [`kth_dataset.py`](video_prediction/datasets/kth_dataset.py) for an example where the original dataset is given as videos.

Note: the `bair` dataset is used for both the action-free and action-conditioned experiments. Set the hyperparameter `use_state=True` to use the action-conditioned version of the dataset.

### Models
- Download the pre-trained models using the following script.
```bash
bash pretrained_models/download_model.sh dataset_name model_name
```
The `dataset_name` should be one of the following: `bair_action_free`, `kth`, or `bair`.
The `model_name` should be one of the available pre-trained models:
- `ours_savp`: our complete model, trained with variational and adversarial losses. Also referred to as `ours_vae_gan`.

The following are ablations of our model:
- `ours_gan`: trained with L1 and adversarial loss, with latent variables sampled from the prior at training time.
- `ours_vae`: trained with L1 and KL loss.
- `ours_deterministic`: trained with L1 loss, with no stochastic latent variables.

See [`pretrained_models/download_model.sh`](pretrained_models/download_model.sh) for a complete list of available pre-trained models.

## Citation

If you find this useful for your research, please use the following.

```
@article{lee2018savp,
  title={Stochastic Adversarial Video Prediction},
  author={Alex X. Lee and Richard Zhang and Frederik Ebert and Pieter Abbeel and Chelsea Finn and Sergey Levine},
  journal={arXiv preprint arXiv:1804.01523},
  year={2018}
}
```
