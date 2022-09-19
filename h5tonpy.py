import os
import h5py
import numpy as np
from PIL import Image

path = './test/h5/'
save_npy = 'test/npy/'
save_rgb = './test/rgb/'
for i in os.listdir(path):
    f = h5py.File(os.path.join(path, i))
    depth = f['depth'][:]
    rgb = f['rgb'][:]
    rgb = np.transpose(np.array(rgb), (1, 2, 0))
    rgb = Image.fromarray(rgb)
    rgbname = os.path.join(save_rgb, i.split('.')[0] + '.jpg')
    print(rgbname)
    rgb.save(rgbname)
    depth_array = np.array(depth)
    depthname = os.path.join(save_npy, i.split('.')[0] + '.npy')
    np.save(depthname, depth_array)
    