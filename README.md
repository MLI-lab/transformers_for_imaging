# Vision Transformers Enable Fast and Robust Accelerated MRI
This repository provides code for reproducing the results of the paper: [Vision Transformers Enable Fast and Robust Accelerated MRI](https://openreview.net/forum?id=cNX6LASbv6), by Kang Lin and Reinhard Heckel.

The code has been tested for the environment in `requirements.txt`, and builds on the code from [fastMRI](https://github.com/facebookresearch/fastMRI), [ConViT](https://github.com/facebookresearch/convit), and [timm](https://github.com/rwightman/pytorch-image-models).

## Datasets
The experiments from the paper were performed using the [fastMRI dataset](https://fastmri.org/dataset) and the [ImageNet dataset](https://www.image-net.org/index.php).

## Installation
First, install PyTorch for your operating system and CUDA setup from the
[PyTorch website](https://pytorch.org/get-started/).  

Then, install all other dependencies from `requirements.txt`. This can be done, for example, by running
```
pip install -r requirements.txt
```
from the directory where you saved `requirements.txt`. Alternatively, you may run
```
pip install fastmri
pip install timm
```
to obtain the dependencies.

## Usage
The code for reproducing the paper results are provided as Jupyter notebooks: `fastmri_training.ipynb` and `imagenet_pretrain.ipynb`. 

The notebook `fastmri_training.ipynb` handles model training, fine-tuning and evaluation on the fastMRI dataset.
The notebook `imagenet_pretrain.ipynb` provides the code for pre-training our models on the ImageNet dataset. 

You may adjust the hyperparamters according to the descriptions in the paper. Also note that in both notebooks the data directory path has to be specified at the marked places.

In the experiments, we also used a simulated single-coil brain dataset, which has been simulated in the same fashion as fastMRI's single-coil knee dataset. The code to reproduce this dataset is provided in `simulate_singlecoil_from_multicoil.ipynb`.

## Citation
```
@inproceedings{
lin2022vision,
title={Vision Transformers Enable Fast and Robust Accelerated {MRI}},
author={Kang Lin and Reinhard Heckel},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=cNX6LASbv6}
}
```
## License
This repository is [Apache 2.0](https://github.com/MLI-lab/transformers_for_imaging/blob/master/LICENSE) licensed.

