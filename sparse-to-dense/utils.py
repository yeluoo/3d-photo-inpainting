import os
import cv2
import torch
import criteria
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from metrics import AverageMeter, Result
from dataloaders.nyu_dataloader import NYUDataset
from dataloaders.kitti_dataloader import KITTIDataset
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo

cmap = plt.cm.viridis

def dataLoader(is_train, 
                root_path, 
                dataset_names, 
                max_depth, 
                sparsifier_names, 
                num_samples, 
                modality, 
                batch_size, 
                workers, ):
    
    train_path = os.path.join(root_path, dataset_names, "train")
    val_path   = os.path.join(root_path, dataset_names, "val")

    max_depth = max_depth if max_depth >= 0.0 else np.inf

    if is_train == "train":
        if sparsifier_names == "UniformSampling":
            sparsifier_names = UniformSampling(num_samples=num_samples, max_depth=max_depth)
        elif sparsifier_names == "SimulatedStereo":
            sparsifier_names = SimulatedStereo(num_samples=num_samples, max_depth=max_depth)

        if dataset_names == "nyudepthv2":
            train_dataset = NYUDataset(train_path, type='train',modality=modality, sparsifier=sparsifier_names)
            val_dataset   = NYUDataset(val_path,   type='val',modality=modality, sparsifier=sparsifier_names)
        elif dataset_names == 'kitti':
            train_dataset = KITTIDataset(train_path, type='train',modality=modality, sparsifier=sparsifier_names)
            val_dataset   = KITTIDataset(val_path,   type='val',modality=modality, sparsifier=sparsifier_names)
        else:
            raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

        return train_loader, val_loader
    else:
        sparsifier_names = None
        test_dataset  = NYUDataset(val_path,   type='test',modality=modality, sparsifier=sparsifier_names)  
        test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) 

        return test_loader 
    

def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def strentch_img(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    return rgb, depth_input_col, depth_target_col, depth_pred_col

def resize_depth(depth, width, height):
    """
    depth : numpy
    width : image of width
    height: image of height
    return: depth
    """
    depth = cv2.blur(depth, (3, 3))
    depth = cv2.resize(depth, (width, height), cv2.INTER_AREA)

    return depth

def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]
    unit_scale = 384.

    if width_orig > height_orig:
        scale = width_orig / unit_scale
    else:
        scale = height_orig / unit_scale

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")
    depth = cv2.blur(depth.numpy(), (3, 3))
    depth_resized = cv2.resize(
        depth, (width, height), interpolation=cv2.INTER_AREA
    )

    return depth_resized

def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))
        
    return 


def save_img(filename, input, pred, target):
    """input: numpy"""
    rgb = input[:,:3,:,:]
    depth = input[:,3:,:,:]
    rgb, depth, target, pred = strentch_img(rgb, depth, target, pred)
    img_merge = np.hstack([rgb, depth, target, pred])
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename + '.png')

    return img_merge

