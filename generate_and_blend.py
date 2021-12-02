#%%
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tqdm import tqdm, trange
import os
from scipy.ndimage import binary_erosion
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from skimage.restoration import inpaint
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

def blend_texture_and_synthetic_nodule(lesion_exp, text, THRESH0 = 0.04, THRESH1 = 0.025):

    mask_lesion_exp = lesion_exp > THRESH0
    blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
    mask_lesion_eroded = (mask_lesion_exp).astype(int) - (binary_erosion(mask_lesion_exp) ).astype(int)
    mask_inpaint = (lesion_exp < THRESH1) *(lesion_exp>0)
    blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
    blend_inpain = inpaint.inpaint_biharmonic(blend, mask_inpaint)
    return blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint

# %% SELCT ONE MODEL
path_models = 'model_weights/ca_nodules_generated_center'
models = os.listdir(path_models)
model_chosen = models[50].split(".index")[0]
ca = CAModel3D()
ca.load_weights(f'{path_models}/{model_chosen}')
models[0]

#%% MATCH THE MODEL SELECTED TO THE NODULES AVAILABLE
names = np.load('names_subset.npy')
nodules = np.load('nodules_demo.npz')
nodules = nodules.f.arr_0
nodules = np.squeeze(nodules)
model_chosen = model_chosen.split('weights_')[-1]
model_idx = np.where(names == model_chosen)[0][0]
nodule_orig = nodules[model_idx][15]

#%% NODULE SYNTHESIS
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

#%% LOAD SYNTHETIC TEXTURE
texture = np.load('data/texture_lung_synthetic_ep4k.npy')
texture = [i+np.abs(np.min(i)) for i in texture]
tt = 40
y_start = 44
x_start = 30
text = texture[0][y_start:y_start+tt,x_start:x_start+tt]

fig ,ax  = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(texture[0], vmin=0, vmax=1)
ax[1].imshow(text, vmin=0, vmax=1)

#%%
GEN = 70
ndl_orig_exp = np.pad(nodules[model_idx][16], ((4,4),(4,4)))
# ndl_orig_exp = nodules[model_idx][16]
# lesion_exp = np.pad(nodule_growing[GEN,16,...], ((4,4),(4,4)))
lesion_exp = nodule_growing[GEN,20,...]
blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint = blend_texture_and_synthetic_nodule(lesion_exp, text)

#%%
fig ,ax  = plt.subplots(2,4, figsize=(16,8))
for idx, i in enumerate([ndl_orig_exp, lesion_exp, mask_lesion_exp, mask_lesion_eroded, 
                         mask_inpaint, blend, blend_inpain, blend_inpain]):
    ax.flat[idx].imshow(i)
    ax.flat[idx].axis('off')
    if idx==6:
        ax.flat[idx].imshow(mask_inpaint, alpha=.3)
fig.tight_layout()

#%%
# ndl_orig_exp = np.pad(nodules[model_idx][20], ((4,4),(4,4)))
ndls_generated = []
for GEN in np.arange(20,50):
    lesion_exp = nodule_growing[GEN,20,...]
    # lesion_exp = np.pad(nodule_growing[GEN,20,...], ((4,4),(4,4)))
    blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint = blend_texture_and_synthetic_nodule(lesion_exp, text)
    ndls_generated.append(blend_inpain)

#%%
fig, ax = plt.subplots(5,6, figsize=(16,12))
for idx, i in enumerate(ndls_generated):
    ax.flat[idx].imshow(i)
    ax.flat[idx].axis('off')
fig.tight_layout()

#%%

#%%

#%%
mask_lesion_exp = lesion_exp > 0.04
blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
mask_lesion_eroded = (mask_lesion_exp).astype(int) - (binary_erosion(mask_lesion_exp) ).astype(int)
mask_inpaint = (lesion_exp <  0.025) *(lesion_exp>0)
blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
blend_inpain = inpaint.inpaint_biharmonic(blend, mask_inpaint)

fig ,ax  = plt.subplots(2,4, figsize=(16,8))
for idx, i in enumerate([ndl_orig_exp, lesion_exp, mask_lesion_exp, mask_lesion_eroded, 
                         mask_inpaint, blend, blend_inpain, blend_inpain]):
    ax.flat[idx].imshow(i)
    ax.flat[idx].axis('off')
    if idx==6:
        ax.flat[idx].imshow(mask_inpaint, alpha=.3)
fig.tight_layout()


#%%

# %%
print(models[50].split('.index')[0])
names = np.load('names_subset.npy')
nodules = np.load('nodules_demo.npz')
nodules = nodules.f.arr_0
names[:3], nodules.shape
nodules = np.squeeze(nodules)

#%%
fig ,ax  = plt.subplots(4,8, figsize=(24,12))
for i in range(32):
    ax.flat[i].imshow(nodules[i,...,15])
    ax.flat[i].axis('off')
fig.tight_layout()

#%%
models_unique = [i for i in models if 'index' in i]
print(len(models_unique), len(names), len(nodules), nodules.shape)

#%%
model_chosen = models[50].split(".index")[0]
model_chosen = model_chosen.split('weights_')[-1]
model_chosen

#%%
model_idx = np.where(names == model_chosen)[0][0]
print(model_idx)
fig ,ax  = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(nodules[model_idx][15])
ax[1].imshow(nodule_growing[200,15,...])

#%%
target = nodules[model_idx]
target_mask = nodules[model_idx]>0
plt.imshow(target_mask)

#%%
MAE = []
for i in nodule_growing:
    MAE.append(np.mean(np.abs(target[15] - (i[15] * target_mask[15]))))
plt.plot(MAE)

#%%
plt.plot(MAE[:200])

#%%
i=70
ndl_orig_exp = np.pad(nodules[model_idx][15], ((4,4),(4,4)))
lesion_exp = np.pad(nodule_growing[i,15,...], ((4,4),(4,4)))
# mask_lesion_exp = np.pad(nodule_growing[i,15,...] > 0.03, ((4,4),(4,4)))
mask_lesion_exp = lesion_exp > 0.04
blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
mask_lesion_eroded = (mask_lesion_exp).astype(int) - (binary_erosion(mask_lesion_exp) ).astype(int)
mask_inpaint = (lesion_exp<0.025) *(lesion_exp>0)
# ndl_inpain = inpaint.inpaint_biharmonic(lesion_exp, mask_inpaint)
blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
blend_inpain = inpaint.inpaint_biharmonic(blend, mask_inpaint)

fig ,ax  = plt.subplots(2,4, figsize=(16,8))
for idx, i in enumerate([ndl_orig_exp, lesion_exp, mask_lesion_exp, mask_lesion_eroded, 
                         mask_inpaint, blend, blend_inpain, blend_inpain]):
    ax.flat[idx].imshow(i)
    ax.flat[idx].axis('off')
    if idx==6:
        ax.flat[idx].imshow(mask_inpaint, alpha=.3)
fig.tight_layout()

#%%
print(nodules.shape)
i=10
mask_lesion_exp = np.pad(nodule_growing[i,15,...] > 0.03, ((4,4),(4,4)))
lesion_exp = np.pad(nodule_growing[i,15,...], ((4,4),(4,4)))
blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
plt.imshow(blend)


#%%
mask_lesion_eroded = binary_erosion(mask_lesion_exp)
fig ,ax  = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(mask_lesion_exp)
ax[1].imshow(mask_lesion_eroded)

#%%
# ndl_ = nodule_growing[10,15,...]
fig ,ax  = plt.subplots(1,2, figsize=(12,6))
ax[0].hist(text.flatten());
ax[1].hist(lesion_exp[lesion_exp>0]);

#%%
fig ,ax  = plt.subplots(1,2, figsize=(12,6))
mask_inpaint = (lesion_exp<0.025) *(lesion_exp>0)
ax[0].imshow(mask_inpaint)
ax[1].imshow(lesion_exp>0)

#%%

ndl_inpain = inpaint.inpaint_biharmonic(lesion_exp, mask_inpaint)
blend = ndl_inpain * mask_lesion_exp + text * (1 - mask_lesion_exp)
plt.imshow(blend)

# %%
nodule_growing = []
GROW_ITER = 100
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
texture = np.load('data/texture_lung_synthetic_ep4k.npy')
texture = [i+np.abs(np.min(i)) for i in texture]
print(np.shape(texture))
fig ,ax  = plt.subplots(2,4, figsize=(24,12))
for i in range(4):
    ax[0,i].imshow(texture[i])
    ax[1,i].hist(texture[i].flatten())
plt.tight_layout()

# %%
tt = 40
y_start = 44
x_start = 30
text = texture[0][y_start:y_start+tt,x_start:x_start+tt]

fig ,ax  = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(texture[0], vmin=0, vmax=1)
ax[1].imshow(text, vmin=0, vmax=1)
# %%
ng = nodule_growing[50,...,15]
print(ng.shape)
plt.imshow(ng)
# %%
nodule_growing.shape
# %%
plt.imshow(nodule_growing[-1][...,16])
# %%
middle_slice = np.shape(np.squeeze(nodule_growing[0]))[0]//2 - 1
plt.imshow(nodule_growing[50,15,...])

# %%
