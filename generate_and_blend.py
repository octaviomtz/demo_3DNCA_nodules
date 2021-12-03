import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tqdm import tqdm, trange
import os
from scipy.ndimage import binary_erosion
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from skimage.restoration import inpaint
from model import CAModel3D
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from utils.utils_figs import fig_multiple3D, blend_texture_and_synthetic_nodule
from utils.utils_nca import VideoWriter

def main():
    aa = os.listdir('.')
    # SELECT ONE MODEL
    path_models = 'model_weights/ca_nodules_generated_center'
    models = os.listdir(path_models)
    model_chosen = models[50].split(".index")[0]
    ca = CAModel3D()
    ca.load_weights(f'{path_models}/{model_chosen}')
    models[0]

    # MATCH THE MODEL SELECTED TO THE NODULES AVAILABLE
    names = np.load('names_subset.npy')
    nodules = np.load('nodules_demo.npz')
    nodules = nodules.f.arr_0
    nodules = np.squeeze(nodules)
    model_chosen = model_chosen.split('weights_')[-1]
    model_idx = np.where(names == model_chosen)[0][0]
    nodule_orig = nodules[model_idx][15]

    # NODULE SYNTHESIS
    nodule_growing = []
    GROW_ITER = 100
    CHANNEL_N = 16
    path_video = f'videos/nodule_0.mp4'
    # x = np.zeros([1, 32, 32, 32, CHANNEL_N], np.float32)
    x = np.zeros([1, 40, 40, 40, CHANNEL_N], np.float32)
    # x[..., 16, 16, 16, 1:] = 1.0
    x[..., 20, 20, 20, 1:] = 1.0
    with VideoWriter(path_video) as vid:
        for i in trange(GROW_ITER):
            for ca, xk in zip([ca], x):
                temp = ca(xk[None,...])[0]
                xk[:] = temp
                vid.add(temp.numpy()[20,...,0])
                nodule_growing.append(temp.numpy()[...,0])
    nodule_growing = np.stack(nodule_growing, axis=0)

    # LOAD SYNTHETIC TEXTURE
    texture = np.load('data/texture_lung_synthetic_ep4k.npy')
    texture = [i+np.abs(np.min(i)) for i in texture]
    tt = 40
    y_start = 44
    x_start = 30
    text = texture[0][y_start:y_start+tt,x_start:x_start+tt]

    # BLEND TEXTURE AND SYNTHETIC LESIONS
    GEN = 70
    ndl_orig_exp = np.pad(nodules[model_idx][16], ((4,4),(4,4)))
    # ndl_orig_exp = nodules[model_idx][16]
    # lesion_exp = np.pad(nodule_growing[GEN,16,...], ((4,4),(4,4)))
    lesion_exp = nodule_growing[GEN,20,...]
    blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint = blend_texture_and_synthetic_nodule(lesion_exp, text)

    # NODULE PROGRESSION
    ndls_generated = []
    for GEN in np.arange(20,50):
        lesion_exp = nodule_growing[GEN,20,...]
        # lesion_exp = np.pad(nodule_growing[GEN,20,...], ((4,4),(4,4)))
        blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint = blend_texture_and_synthetic_nodule(lesion_exp, text)
        ndls_generated.append(blend_inpain)

    # FIGURES
    # FIG 1. 
    fig ,ax  = plt.subplots(2,4, figsize=(16,8))
    for idx, i in enumerate([ndl_orig_exp, lesion_exp, mask_lesion_exp, mask_lesion_eroded, mask_inpaint, blend, blend_inpain, blend_inpain]):
        ax.flat[idx].imshow(i)
        ax.flat[idx].axis('off')
        if idx==6:
            ax.flat[idx].imshow(mask_inpaint, alpha=.3)
    fig.tight_layout()
    plt.savefig('results/blend_texture_process.png')
    # FIG 2.
    fig, ax = plt.subplots(5,6, figsize=(16,12))
    for idx, i in enumerate(ndls_generated):
        ax.flat[idx].imshow(i)
        ax.flat[idx].axis('off')
    fig.tight_layout()
    plt.savefig('results/nodule_progression.png')

if __name__ == '__main__':
    main()
