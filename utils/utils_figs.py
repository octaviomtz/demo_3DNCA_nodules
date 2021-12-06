import numpy as np
import matplotlib.pylab as plt
from skimage.restoration import inpaint
import imageio
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

def create_list_with_blended_nodules(nodule_growing, text, GROW_ITER = 100):  
    ndls_generated = []
    gens = []
    for GEN in np.arange(0, GROW_ITER):
      lesion_exp = nodule_growing[GEN,20,...]
      blend_inpain, blend, mask_lesion_exp, mask_lesion_eroded, mask_inpaint = blend_texture_and_synthetic_nodule(lesion_exp, text)
      ndls_generated.append(blend_inpain)
      gens.append(GEN)
    return ndls_generated, gens

def fig_list_with_blended_nodules(ndls_generated):
    fig, ax = plt.subplots(4,10, figsize=(20,8))
    for idx, i in enumerate(ndls_generated):
        ax.flat[idx].imshow(i, vmin=0, vmax=1)
        ax.flat[idx].axis('off')
    fig.tight_layout()
    plt.savefig('results/nodule_growing.png')

def fig_for_gif(ndls_generated, gens, bar_gif, path_imgs_for_gif='results/imgs_for_gif/img_temp.png', path_gif='results/nodule_growing.gif'):
    images = []
    step = 1 / len(ndls_generated)
    for idx, (i, gen) in enumerate(zip(ndls_generated, gens)):
        bar_gif.progress((idx+1) * step)
        fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [30, 1]}, figsize=(4,4));
        ax[0].imshow(i, vmin=0, vmax=1)
        ax[0].text(4,4,gen, color='r')
        ax[0].axis('off')
        ax[1].vlines(x=0, ymin=0, ymax=gens[-1], color='k');
        ax[1].scatter(0,gen, c='k', s=gens[-1]);
        ax[1].set_ylim([0,gens[-1]]);
        ax[1].text(0,0, 1, fontsize=12, c='k')
        ax[1].text(0,gens[-1]-5, gens[-1]+1, fontsize=12, c='k')
        ax[1].axis('off')
        fig.tight_layout();
        plt.savefig(path_imgs_for_gif)
        images.append(imageio.imread(path_imgs_for_gif));
        plt.close()
    imageio.mimsave(path_gif, images, fps=4)
