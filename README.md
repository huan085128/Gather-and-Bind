# Enhancing Semantic Mapping in Text-to-Image Diffusion via Gather-and-Bind

This repo is the official PyTorch implementation of [Gather-and-Bind](https://doi.org/10.1016/j.cag.2024.104118).

![Local Image](./images/4.png)

## Project Introduction

Gather-and-Bind is an optimization algorithm for stable diffusion, specifically designed to enhance the quality of generated images to make them align better with the given prompts.

<div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
        <img src="images/3.png" width="400" alt="Image 1">
    </div>
    <div>
        <img src="images/2.png" width="400" alt="Image 2">
    </div>
</div>

![Gather-and-Bind](images/1.png)

## Get Start

### Step-1 installation
To set up and run this project, please ensure the following requirements are met:
- Conda is mandatory: The installation and environment setup rely on Conda. Please install Miniconda or Anaconda if you haven’t already.
- Hardware requirements: A GPU with at least 24GB VRAM is recommended to run the model efficiently. 

```bash
# clone the repository
git clone https://github.com/huan085128/Gather-and-Bind

# install environment
./install.sh
```
if you can't download en_core_web_trf, you can download it manually from [here](https://drive.google.com/drive/folders/1A_66w8pqR9JnZxpy4Pz1Eyuh1IH4I4Zu?usp=sharing) and then install it with the following command:

```bash
python install en_core_web_trf-3.7.3.tar.gz
```

### Step-2 download pretrained model
Download the stable_diffusion_v1.5 model from [here](https://drive.google.com/drive/folders/1A_66w8pqR9JnZxpy4Pz1Eyuh1IH4I4Zu?usp=sharing)

and then put the model in the models directory.

### Step-3 generate images

```bash
# run the following command to generate images
python generate_images.py
```

### Step-4 visualizing attention maps with jupyter notebook

As part of the experiments in our paper, we provide a Jupyter Notebook that visualizes the changes in attention maps during the denoising process of the diffusion model. This notebook allows for a more intuitive understanding of how the model focuses on different parts. 

please see the `explain.ipynb` file for more details.
