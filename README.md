# Demo: Lung Nodule Nodule Synthesis using Neural Cellular Automata

## Overview.
In a previous study we use inpainting and NCA to synthetize lung nodules (https://ieeexplore.ieee.org/document/9433893). Here, we generate the healthy tissue on the background as an alternative approach.   
This work is based on the NCA developed by _**Mordvintsev A.**, et al., "Growing Neural Cellular Automata", Distill, 2020._

## API demo
![api_demo](/github_images/gif_nodule_synthesis_api.gif?raw=true)

## Create healthy background
![texture_and_mask_lung](/github_images/texture_and_mask_lung.png?raw=true) 
First we need to collect images of healthy lung tissue (without lung nodules or covid lesions):   
1. We use the script texture_mosaic.py to select the largest rectangle from an area inside the lungs that does not contain any lesion.   
1. Then we use a bin packing algorithm to create a mosaic of all these  rectangles.   
```bash
python texture_mosaic.py data_folder='path_to_lung_dataset' SCAN_NAME='all' n_scans=60 # to go through a dataset   
python texture_mosaic.py #to only go through one scan (won't produce enough rectangles)
```
1. The previous step will produce a mask that can be used to inpaint the boundaries of the rectangles and the gaps between them.   
1. Apply inpainting with deep image prior (misc/inpainting.ipynb) to the mosaic using its mask.  

## Generate synthetic texture

## Train 3D NCA
1. Use the train_synthesis.py to train a 3D NCA to generate a nodule 