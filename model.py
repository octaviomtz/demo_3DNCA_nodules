import tensorflow as tf
from tensorflow.keras.layers import Conv3D
import numpy as np
from utils_nca import get_living_mask

class CAModel3D(tf.keras.Model):

  def __init__(self, channel_n=16, fire_rate=0.5):
    super().__init__()
    self.channel_n = channel_n
    self.fire_rate = fire_rate

    self.dmodel = tf.keras.Sequential([
          Conv3D(128, 1, activation=tf.nn.relu),
          Conv3D(self.channel_n, 1, activation=None,
              kernel_initializer=tf.zeros_initializer),
    ])

    self(tf.zeros([1, 3, 3, 3, channel_n]))  # dummy call to build the model

  @tf.function
  def perceive(self, x, angle=0.0):
    identify = np.float32(np.zeros((3,3,3)))
    identify[1,1,1] = 1
    dx = np.float32(np.asarray([[[1,2,1],[2,4,2],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]]) / 32.0) # Engel. Real-Time Volume Graphics (p.112)
    dy = dx.transpose((1,0,2))
    dz = dx.transpose((2,1,0))
    c, s = tf.cos(0.), tf.sin(0.)
    kernel = tf.stack([identify, dx, dy, dz], -1)[:, :, :, None, :] # we removed the sin cos used for rotations
    kernel = tf.repeat(kernel, self.channel_n, 3) # OMM WARNING maybe the last param is 2
    # kernel = tf.repeat(kernel, self.channel_n, 2)
    # function to replace tf.nn.depthwise_conv3d
    # https://github.com/alexandrosstergiou/keras-DepthwiseConv3D/blob/master/DepthwiseConv3D.py
    input_dim = x.shape[-1]
    groups = input_dim
    y = tf.concat([tf.nn.conv3d(x[:,:,:,:,i:i+input_dim//groups], kernel[:,:,:,i:i+input_dim//groups,:],
                    strides=[1, 1, 1, 1, 1],
                    padding= 'SAME') 
    for i in range(0,input_dim,input_dim//groups)], axis=-1)
    
    return y

  @tf.function
  def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
    pre_life_mask = get_living_mask(x)

    y = self.perceive(x, angle)
    dx = self.dmodel(y)*step_size
    if fire_rate is None:
      fire_rate = self.fire_rate
    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :, :1])) <= fire_rate
    x += dx * tf.cast(update_mask, tf.float32)

    post_life_mask = get_living_mask(x)
    life_mask = pre_life_mask & post_life_mask
    return x * tf.cast(life_mask, tf.float32)
