#%%
import os
import io
import PIL.Image, PIL.ImageDraw
import zipfile
import json
import numpy as np
import glob
import time
import tensorflow as tf
from tqdm import tqdm
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'

import matplotlib.pyplot as plt
import sys

from utils.utils_figs import fig_multiple3D
from model import CAModel3D
from utils.utils_nca import get_center_of_volume_from_largest_component_return_center_or_extremes
from utils.utils_nca import loss_f, export_model, to_rgba

# %%
nodules3D = np.load('nodules_demo.npz')
nodules3D = nodules3D.f.arr_0
print(nodules3D.shape)
fig_multiple3D(nodules3D[10:],r=4,c=4,name='nodules 1400')

# %%
CHANNEL_N = 16   
TARGET_PADDING = 16 
TARGET_SIZE = 40
BATCH_SIZE = 4
POOL_SIZE = 64 # WARNING changed from 1024 to 64
CELL_FIRE_RATE = 0.5

USE_PATTERN_POOL = 0
DAMAGE_N = 0

TRAIN_EPOCHS = 1000
NDL_OFFSET = 452
NDL_NUM = 23

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

# %%
CAModel3D().dmodel.summary()

#%%
idx_rand = np.random.randint(0, len(nodules3D))
ndl = nodules3D[idx_rand]

# 1. GET NODULE TARGET
FISH = (32,32,32,2) # final shape
filename_one_nodule = f'nodule_{idx_rand:03d}'
nsh = np.shape(ndl)
ndl_pad_temp = np.pad(np.squeeze(ndl),(((FISH[0] - nsh[0])//2, (FISH[0] - nsh[0])//2), ((FISH[1] - nsh[1])//2, (FISH[1] - nsh[1])//2), ((FISH[2] - nsh[2])//2, (FISH[2] - nsh[2])//2)))
target_img = np.zeros(FISH)
target_img[:,:,:,0] = ndl_pad_temp
target_img[:,:,:,1] = ndl_pad_temp>0
filename_one_nodule

# %%
# 2. GET THE SEED
#print(f'2. {one_nodule}')
h, w, d = target_img.shape[:3]
seed = np.zeros([h, w, d, CHANNEL_N], np.float32)
iq, jq, kq = get_center_of_volume_from_largest_component_return_center_or_extremes(np.squeeze(ndl_pad_temp), ndl, extreme=1, coord='coord2')
seed[iq, jq, kq , 1:] = 1.0
x = np.expand_dims(seed,0)

#%%
# 3. MODEL, LOSS AND LEARNER
tf.compat.v1.reset_default_graph()  
ca = CAModel3D()
loss_log = []
loss0 = loss_f(seed, target_img).numpy()
!mkdir -p train_logA && rm -f train_logA/*

#%%
def loss_f(x):
  return tf.reduce_mean(tf.square(to_rgba(x)-target_img), [-2, -3, -4, -1])

#%% 4. TRAIN
@tf.function
def train_step(x):
    iter_n = tf.random.uniform([], 64, 96, tf.int32) # random range
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            x = ca(x)
        loss = tf.reduce_mean(loss_f(x))
    grads = g.gradient(loss, ca.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss

#%%
TRAIN_EPOCHS=100
for i in tqdm(range(TRAIN_EPOCHS+1), desc = str(idx_rand)):
    x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x, loss = train_step(x0)
    loss_log.append(loss.numpy())
    if i in [TRAIN_EPOCHS]:
        export_model(ca, './train_logA/%04d'%i, CHANNEL_N)

#%% 5. GET TRAINED IMAGES
grow_iterations = 100
models = []
for i in [TRAIN_EPOCHS]: # 
    ca = CAModel3D()
    ca.load_weights('./train_logA/%04d'%i)
    models.append(ca)
nodule_growing = []
x = np.zeros([len(models), 32, 32, 32, CHANNEL_N], np.float32)
x[..., iq, jq, kq, 1:] = 1.0
for i in range(grow_iterations):
    for ca, xk in zip(models, x):
        temp = ca(xk[None,...])[0]
        xk[:] = temp
        nodule_growing.append(temp.numpy()[...,0]) 

#%% 6. SAVE NODULE FROM MODEL TRAINED THE LONGEST
#print(f'6. {one_nodule}')
if len(models)>1:
    nodule_growing = [i for idx, i in enumerate(nodule_growing) if idx%len(models)-1==0] # get the images from the last model
nodule_growing = np.stack(nodule_growing, axis=0) # convert from list of 3d arrays to 4d array
np.savez_compressed(f'model_weights/{filename_one_nodule}.npz', nodule_growing)
loss_log_py = np.asarray(loss_log)
np.save(f'model_weights/{filename_one_nodule}.npy', loss_log_py)
ca.save_weights(f'model_weights/{filename_one_nodule}')