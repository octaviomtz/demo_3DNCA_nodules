import numpy as np
import matplotlib.pylab as plt
from skimage.restoration import inpaint
from scipy.ndimage import binary_erosion

def figure_4_channels(ff, name='ff'):
    fig, ax = plt.subplots(1,4, figsize=(8,2))
    for idx in range(4):
        ax[idx].imshow(ff[:,:,idx])
    for axx in ax.ravel(): axx.axis('off')
    fig.suptitle(f'{name} {np.shape(ff)}')

def fig_multiple(ff, r=1, c=4, name='ff', vmin=0, vmax=1, figmul=1):
  fig, ax = plt.subplots(r,c, figsize=(c*2*figmul,r*2*figmul))
  for idx in range(r*c):
    if r <= 1:
      ax[idx].imshow(ff[idx])
    else:
      # print(idx,idx//c,idx%c)
      ax[idx//c,idx%c].imshow(ff[idx], vmin=0, vmax=1)
  for axx in ax.ravel(): axx.axis('off')
  fig.suptitle(f'{name}')
  fig.tight_layout()

def fig_multiple3D(ff, r=1, c=4, name='ff', vmin=0, vmax=1, cmap='viridis'):
  '''Plot the middle slice of a 3D array.
  We assume that np.shape(ff) == B'''
  middle_slice = np.shape(np.squeeze(ff[0]))[0]//2 - 1
  fig, ax = plt.subplots(r,c, figsize=(c*2,r*2))
  for idx in range(r*c):
    if r <= 1:
      ax[idx].imshow(np.squeeze(ff[idx])[middle_slice,...], cmap=cmap)
    else:
      ax[idx//c,idx%c].imshow(np.squeeze(ff[idx])[middle_slice,...], vmin=0, vmax=1, cmap=cmap)
  for axx in ax.ravel(): 
      axx.axis('off')
  fig.tight_layout()

def print_3d_horiz(dx):
  '''https://stackoverflow.com/questions/58365744/python-3d-array-printing-horizontally'''
  j=0
  for i in range(len((dx[j]))):
      for j in range(len(dx)):        
              for k in range(len((dx[j][i]))):  
                  if k == 2: print(f'{dx[j][i][k]:+d}  ',end = ' ')
                  else: print(f'{dx[j][i][k]:+d}',end = ' ')
      print()

def blend_texture_and_synthetic_nodule(lesion_exp, text, THRESH0 = 0.04, THRESH1 = 0.025):
    mask_lesion_exp = lesion_exp > THRESH0
    blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
    mask_lesion_eroded = (mask_lesion_exp).astype(int) - (binary_erosion(mask_lesion_exp) ).astype(int)
    mask_inpaint = (lesion_exp < THRESH1) *(lesion_exp>0)
    blend = lesion_exp * mask_lesion_exp + text * (1 - mask_lesion_exp)
    blend_inpain = inpaint.inpaint_biharmonic(blend, mask_inpaint)
    return blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint