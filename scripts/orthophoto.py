import os, json
import numpy as np, matplotlib.pyplot as plt

orthophotos = list()
coords = list()
op_dir = r'data/test_ortophoto/images'
for file in os.listdir(op_dir):
        coords.append(np.array(file.replace('.png', '').split(sep='_'), dtype=np.float64))
        file = op_dir+'/' + file
        orthophotos.append(plt.imread(file))
        
coords = np.array(coords).reshape(len(coords), 4) 
orthophotos= np.array(orthophotos).reshape(len(orthophotos),500, 500, 4) 
coords = coords[np.argsort(coords[:, 0])]
orthophotos = orthophotos[np.argsort(coords[:, 0])]

xmin, xmax, ymin, ymax = np.min(coords[:, 0]),\
                         np.max(coords[:, 2]),\
                         np.min(coords[:, 1]),\
                         np.max(coords[:, 3])

im_coords = coords.copy()
im_coords[:, 0] -= xmin
im_coords[:, 2] -= xmin
im_coords[:, 1] -= ymin
im_coords[:, 3] -= ymin
im_coords = (im_coords/100*500).astype('int32')

nx = np.max(np.unique(coords[:, 0], return_counts=True)[1])
ny = np.max(np.unique(coords[:, 1], return_counts=True)[1])
orthophoto_full = np.zeros((nx*500, ny*500, 4))
for im_coord, im in zip(im_coords, orthophotos):
    orthophoto_full[im_coord[1]:im_coord[3], im_coord[0]:im_coord[2]] = im[::-1,]

orthophoto_full = orthophoto_full[::-1]
corner_coords = np.array((xmin, xmax, ymin, ymax), dtype='int64')
np.save(r'Code/testing/ims/orthophoto/orthophoto', orthophoto_full)
np.save(r'Code/testing/ims/orthophoto/orthophoto_coords', corner_coords)
      