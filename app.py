import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from model import CAModel3D
from utils.utils_figs import create_list_with_blended_nodules, fig_list_with_blended_nodules, fig_for_gif
from utils.utils_app import match_models_and_nodules, grow_nodule, load_texture

# warnings.filterwarnings('ignore')

nodules = np.load('nodules_demo.npz')
nodules = nodules.f.arr_0
path_models = 'model_weights/ca_nodules_generated_center'
if 'dict_match' not in locals():
    print('matching models and nodules')
    dict_match, match_nodule = match_models_and_nodules(path_models)


st.markdown('## Nodule synthesis with Cellular automata')
st.markdown('## Original Nodules')
st.markdown("Select nodule to recreate its emergence and growing.")
st.image('github_images/nodules_with_trained_model.png', use_column_width=True)

IDX = st.slider(key='NODULE', label="Pick a nodule", min_value=0, max_value=49, step=1, value = 21)
col00, col01, col02 = st.columns((1, 1, 1))

synthetic_texture = col00.button('Change synthetic texture ')
apply_model = col00.button('Grow Nodule. Make images')
nodule = nodules[match_nodule[IDX]]

model_chosen = dict_match[match_nodule[IDX]].split(".index")[0]
print(model_chosen)

plt.figure(figsize=(2,2))
plt.imshow(np.squeeze(nodule[15]), vmin=0, vmax=1)
plt.axis('off')
plt.savefig('results/nodule_chosen.png') 
col01.image('results/nodule_chosen.png')
st.session_state['test']=1
if 'change_synthetic' not in st.session_state: 
    col02.image('results/texture_mini.png') 

# text_box = st.empty()
# text_box.text_area('text', IDX)
my_bar = col00.progress(0)
bar_gif = col00.progress(0)

if synthetic_texture:
    col02.empty()
    st.session_state['change_synthetic'] = 1
    y_rand = np.random.randint(0, 80)
    x_rand = np.random.randint(0, 80)
    texture = load_texture(y_start = y_rand, x_start = x_rand)
    st.session_state['texture'] = texture
    
    col02.image('results/texture_mini.png')

if apply_model:
    ca = CAModel3D()
    ca.load_weights(f'{path_models}/{model_chosen}')
    nodule_growing = grow_nodule(ca, my_bar, GROW_ITER = 100)
    pad=4
    nodule_padded = np.pad(np.squeeze(nodule),((pad,pad),(pad,pad),(pad,pad)))
    # text_box.text_area(f'{len(nodule_growing), np.shape(nodule_growing), np.shape(nodule_padded), np.min(nodule_padded), np.max(nodule_padded)}')
    ndls_generated, gens = create_list_with_blended_nodules(nodule_growing, st.session_state['texture'], nodule_padded)
    # fig_list_with_blended_nodules(ndls_generated)
    # st.image('results/nodule_growing.png')    
    fig_for_gif(ndls_generated, gens, bar_gif)
    st.image('results/nodule_growing.gif')
# st.sidebar.markdown("### Background")
# st.sidebar.markdown("Select nodule to recreate its emergence and growing.")
