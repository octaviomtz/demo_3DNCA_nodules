#%%
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tqdm import tqdm, trange
import os

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from model import CAModel3D
from utils.utils_figs import fig_multiple3D

#%%
class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

# %%
path_models = 'model_weights/ca_nodules_generated_center'
models = os.listdir(path_models)
models[0]

#%%
ca = CAModel3D()
ca.load_weights(f'{path_models}/{models[50].split(".index")[0]}')

# %%
models[50].split('.index')[0]

# %%
nodule_growing = []
GROW_ITER = 500
CHANNEL_N = 16
path_video = f'videos/nodule_0.mp4'
x = np.zeros([1, 32, 32, 32, CHANNEL_N], np.float32)
x[..., 16, 16, 16, 1:] = 1.0
with VideoWriter(path_video) as vid:
    for i in trange(GROW_ITER):
        for ca, xk in zip([ca], x):
            temp = ca(xk[None,...])[0]
            xk[:] = temp
            vid.add(temp.numpy()[15,...,0])
            nodule_growing.append(temp.numpy()[...,0])
nodule_growing = np.stack(nodule_growing, axis=0)

# %%
fig_multiple3D(nodule_growing,5,10)

#%%
temp.shape
# %%
