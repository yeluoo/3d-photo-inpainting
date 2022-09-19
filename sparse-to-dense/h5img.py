import os
import tqdm
import h5py
import numpy as np
from PIL import Image


path = "./inputs/val/"
save_path = "./inputs/rgb/"
if not os.path.exists(save_path): os.makedirs(save_path)

for i in tqdm.tqdm(os.listdir(path)):
    path1 = os.path.join(path, i)
    for j in os.listdir(path1):
        h5 = h5py.File(os.path.join(path1, j), 'r')
        depth = h5['depth'][:]
        rgb = h5['rgb'][:]
        depth = np.array(depth)
        rgb = np.transpose(np.array(rgb), (1, 2, 0))
        depth = Image.fromarray(depth.astype('uint8'))
        rgb = Image.fromarray(np.uint8(rgb))
        rgb = rgb.resize((640*2, 480*2))
        rgb.save(os.path.join(save_path, os.path.splitext(j)[0]+'.png'))

# classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# print(classes)
# l = [j for i in os.listdir(path) for j in os.listdir(os.path.join(path, i))]
# print(l)