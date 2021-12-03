import numpy as np
import os
from tqdm import tqdm, trange
import matplotlib.pylab as plt
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from utils.utils_nca import VideoWriter

def match_models_and_nodules(path_models):
    models = os.listdir(path_models)
    models = [i for i in models if '.index' in i]
    print(len(models))
    names = np.load('names_subset.npy')
    nodules = np.load('nodules_demo.npz')
    nodules = nodules.f.arr_0
    nodules = np.squeeze(nodules)

    match_model = []; match_nodule = []
    for i in models:
        model_chosen = i.split(".index")[0].split('weights_')[-1]
        model_idx = np.where(names == model_chosen)[0][0]
        match_nodule.append(model_idx)
        match_model.append(i)

    zip_iterator = zip(match_nodule, match_model)
    dict_match = dict(zip_iterator)
    return dict_match, match_nodule

def grow_nodule(ca, GROW_ITER = 100):
    nodule_growing = []
    
    CHANNEL_N = 16
    path_video = f'videos/nodule_0.mp4'
    x = np.zeros([1, 40, 40, 40, CHANNEL_N], np.float32)
    x[..., 20, 20, 20, 1:] = 1.0
    with VideoWriter(path_video) as vid:
        for i in trange(GROW_ITER):
            for ca, xk in zip([ca], x):
                temp = ca(xk[None,...])[0]
                xk[:] = temp
                vid.add(temp.numpy()[20,...,0])
                nodule_growing.append(temp.numpy()[...,0])
    nodule_growing = np.stack(nodule_growing, axis=0)
    return nodule_growing

#%% LOAD SYNTHETIC TEXTURE
def load_texture(path_texture = 'data/texture_lung_synthetic_ep4k.npy', y_start = 44, x_start = 30):
    texture = np.load(path_texture)
    texture = [i+np.abs(np.min(i)) for i in texture]
    tt = 40
    text = texture[0][y_start:y_start+tt,x_start:x_start+tt]
    
    plt.figure(figsize=(3,3))
    plt.imshow(text, vmin=0, vmax=1)
    plt.savefig('results/texture_mini.png')
    return text

