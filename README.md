# Vision Transformers Enable Fast and Robust Accelerated MRI
This repository provides code for reproducing the results of the paper: [Vision Transformers Enable Fast and Robust Accelerated MRI](https://openreview.net/forum?id=cNX6LASbv6), by Kang Lin and Reinhard Heckel.

The code has been tested for the environment given in `requirements.txt`, and builds on the code from [fastMRI](https://github.com/facebookresearch/fastMRI), [ConViT](https://github.com/facebookresearch/convit), and [timm](https://github.com/rwightman/pytorch-image-models).

## Datasets
The experiments from paper were performed using the [fastMRI](https://fastmri.org/dataset) and the [ImageNet](https://www.image-net.org/index.php) dataset.

## Installation
First, install PyTorch for your operating system and CUDA setup from the
[PyTorch website](https://pytorch.org/get-started/).  

Then, install all other dependencies from `requirements.txt`. This can be done, for example, by downloading `requirements.txt`, and running
```
pip install -r requirements.txt
```
in the directory where you saved `requirements.txt`. Alternatively, you may run
```
pip install fastmri
pip install timm
```
to obtain the dependencies.

## Usage
The code to reproduce our paper results are provided as Jupyter notebooks: `fastmri_training.ipynb` and `imagenet_pretrain.ipynb`. 

The notebook `fastmri_training.ipynb` handles model training, fine-tuning and evaluation on the fastMRI dataset.
The notebook `imagenet_pretrain.ipynb` provides the code for pre-training our models on the ImageNet dataset. 

You may adjust the hyperparamters according to the descriptions in the paper. Also note that in both notebooks the data directory path has to be clarified at the marked places.

In the experiments, we also used a simulated single-coil brain dataset, which has been simulated in the same fashion as fastMRI's single-coil knee dataset. The code to reproduce this dataset is provided in `simulate_singlecoil_from_multicoil.ipynb`.

## Citation
```
@article{linVisionTransformersEnable2021,
  title = {Vision Transformers Enable Fast and Robust Accelerated MRI},
  author = {Lin, Kang and Heckel, Reinhard},
  year = {2021},
  langid = {english}
}
```
## License
This repository is [Apache 2.0](https://github.com/MLI-lab/transformers_for_imaging/blob/master/LICENSE) licensed.

