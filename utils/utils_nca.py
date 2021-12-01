import io
import numpy as np
from IPython.display import Image, HTML, clear_output
import PIL
import base64
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

from scipy.ndimage import label
from scipy.ndimage import distance_transform_bf
from scipy.spatial import distance

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

# def imshow(a, fmt='jpeg'):
#   display(Image(data=imencode(a, fmt)))

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

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

#==============


def to_rgba(x):
    return x[..., :2] # OMM modified for 1 channel

def to_alpha(x):
    return tf.clip_by_value(x[..., 1:2], 0.0, 1.0)

def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :1], to_alpha(x)
    return 1.0-a+rgb

def get_living_mask(x):
    # return tf.nn.max_pool2d
    return tf.nn.max_pool3d(to_alpha(x), 3, [1, 1, 1, 1, 1], 'SAME') > 0.1

# def make_seed(size, n=1):
#     x = np.zeros([n, size, size, CHANNEL_N], np.float32)
#     x[:, size//2, size//2, 1:] = 1.0
#     return x

# def imshow(a, fmt='jpeg'):
#     display(Image(data=imencode(a, fmt)))

def to_RGBA(x):
    '''create an image of 4 channels where the first three channels are a copy of 
    the first channel of x and the last channel (alpha) is the last channel of x'''
    # create empty image
    empty_size = list(np.shape(x[...,0])) + list([4])
    empty = np.zeros(empty_size)
    slice0 = x[...,0]
    for i in range(3):
        empty[:,:,i] = to_rgba(x)[...,0]
    empty[:,:,-1] = x[...,-1]
    return empty

#============

def generate_pool_figures(pool, step_i, sz=32):
    tiled_pool = tile2d(np.squeeze(to_rgb(pool.x[:15,16]))) # :49 was changed to :15
    fade = np.linspace(1.0, 0.0, sz)
    ones = np.ones(sz) 
    tiled_pool[:, :sz] += (-tiled_pool[:, :sz] + ones[None, :]) * fade[None, :] 
    tiled_pool[:, -sz:] += (-tiled_pool[:, -sz:] + ones[None, :]) * fade[None, ::-1]
    tiled_pool[:sz, :] += (-tiled_pool[:sz, :] + ones[:, None]) * fade[:, None]
    tiled_pool[-sz:, :] += (-tiled_pool[-sz:, :] + ones[:, None]) * fade[::-1, None]
    imwrite('train_logB/%04d_pool.jpg'%step_i, tiled_pool)

def export_model(ca, base_fn, CHANNEL_N):
    ca.save_weights(base_fn)

    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, None, CHANNEL_N]),
        fire_rate=tf.constant(0.5),
        angle=tf.constant(0.0),
        step_size=tf.constant(1.0))
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
        'format': 'graph-model',
        'modelTopology': graph_json,
        'weightsManifest': [],
    }
    with open(base_fn+'.json', 'w') as f:
        json.dump(model_json, f)

#===========

def largest_distance(ndl):
    aa = np.squeeze(ndl)
    zz,yy,xx = np.where(aa > 0)
    point_cloud = np.asarray([zz,yy,xx]).T
    Y = distance.cdist(point_cloud,point_cloud)
    max_dist, _ = np.where(Y==np.max(Y))
    max_dist[0], max_dist[1]
    coord1 = point_cloud[max_dist[0]]
    coord2 = point_cloud[max_dist[1]]
    return coord1, coord2

def get_center_of_volume_from_largest_component_return_center_or_extremes(vol, ndl, extreme=0, coord='coord1'):
    '''get largest component and calculate center of irregular volume using distance transform and
    return those coords'''
    assert (extreme==0 or extreme==1)
    # get largest component
    mask_multiple, cc_num = label(vol > 0)
    sorted_comp = np.bincount(mask_multiple.flat)
    sorted_comp = np.sort(sorted_comp)[::-1]
    comp_largest = (mask_multiple == np.where(np.bincount(mask_multiple.flat) == sorted_comp[1])[0][0])
    if extreme == 0:
        # calculate center of irregular volume using distance transform
        center_volume = distance_transform_bf(comp_largest.astype('float'))
        center_z,center_y,center_x = np.where(center_volume == np.max(center_volume))
        center_z,center_y,center_x = center_z[0],center_y[0],center_x[0]
    if extreme == 1:
        coord1, coord2 = largest_distance(ndl)
        if coord == 'coord2':
            center_z,center_y,center_x = coord2
        else:
            center_z,center_y,center_x = coord1

    return center_z,center_y,center_x

def loss_f(x, target_img):
  return tf.reduce_mean(tf.square(to_rgba(x)-target_img), [-2, -3, -4, -1])