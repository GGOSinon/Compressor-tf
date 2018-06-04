from PIL import Image, ImageOps
import fnmatch
import os
import copy
import numpy as np
import math
from skimage.measure import compare_ssim as ssim

def get_size(img_path):
    Img = Image.open(img_path)
    return Img.size

def normalize_img(img):
    return img/255.

def denormalize_img(img):
    return (img*255).astype(np.uint8)

def bound_img(img):
    img_fin = img
    img_fin = np.maximum(img_fin, 0.)
    img_fin = np.minimum(img_fin, 1.)
    img_fin = denormalize_img(img_fin)
    img_fin = normalize_img(img_fin)
    return img_fin

def get_cord(sz, mx_skip, slice_sz):
    if sz<=slice_sz: return [0]
    dsz = slice_sz - mx_skip
    H = (sz-dsz-1)//mx_skip+1
    res = []
    res.append(0)
    for i in range(1, H-1):
        res.append((i*(sz-dsz))//H)
    res.append(sz-slice_sz)
    return res

def merge_image(img, img_size, max_skip, slice_size):
    dsz = (slice_size[0] - max_skip[0])//2, (slice_size[1]- max_skip[1])//2
    hlist = get_cord(img_size[0], max_skip[0], slice_size[0])
    wlist = get_cord(img_size[1], max_skip[1], slice_size[1])
    print(img.shape, (img_size[1], img_size[0]), max_skip, slice_size, hlist, wlist)
    H, W = len(hlist), len(wlist)
    img = np.reshape(img, (H, W, slice_size[0], slice_size[1]))
    new_img = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
    chk_img = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
    for i in range(W):
        if W==1: range_w = (0, img_size[1])
        elif i==0: range_w = (0, slice_size[0]-dsz[0])
        elif i==W-1: range_w = (dsz[0], slice_size[0])
        else: range_w = (dsz[0], slice_size[0]-dsz[0])
        for j in range(H):
            if H==1: range_h = (0, img_size[0])
            elif j==0: range_h = (0, slice_size[1]-dsz[1])
            elif j==H-1: range_h = (dsz[1], slice_size[1])
            else : range_h = (dsz[1], slice_size[1]-dsz[1])
            for k in range(range_w[0], range_w[1]):
                kk = wlist[i] + k
                for l in range(range_h[0], range_h[1]):
                    ll = hlist[j] + l
                    #print(kk, ll)
                    new_img[kk][ll]+=img[j][i][k][l]
                    chk_img[kk][ll]+=1

    for i in range(img_size[1]):
        for j in range(img_size[0]):
            new_img[i][j]/=chk_img[i][j]

    print(chk_img)
    return new_img

def get_jpeg_from_image(img, qf = 10):
    print(img.shape)
    img = denormalize_img(img)
    h, w = img.shape[0], img.shape[1]
    img = np.reshape(img, (h, w))
    res_img = []
    img = Image.fromarray(img, 'L')
    img.save('temp.jpeg', quality=qf)
    img = Image.open('temp.jpeg')
    img = image_to_np(img)
    print(img.size)
    os.remove('temp.jpeg')
    return img

def np_to_image(img_np):
    img = np.reshape(img_np, (img_np.shape[0], img_np.shape[1]))
    img = np.maximum(img, 0.)
    img = np.minimum(img, 1.)
    img = denormalize_img(img)
    img = Image.fromarray(img, 'L')
    return img

def image_to_np(Img):
    #w, h = Img.size[0], Img.size[1]
    #image_np = np.fromstring(Img.tobytes(), dtype=np.uint8).astype(float)
    #print(image_np.shape)
    #image_np = image_np.reshape((w, h, 1))
    #image_np = normalize_img(image_np)
    image_np = np.array(Img)
    #print(image_np.max())
    image_np = np.reshape(image_np, (image_np.shape[0], image_np.shape[1], -1))
    image_np = normalize_img(image_np)
    return image_np

def make_data_with_path(image_path, isTest = False):
    image = Image.open(image_path)
    new_image_list = []
    h = image.size[0]
    w = image.size[1]
    if len(image.tobytes()) == h*w: is_gray = True
    else: is_gray = False
    if is_gray==False: image = ImageOps.grayscale(image)
    return image_to_np(image)

def split_image(image, max_skip, slice_size):
    new_image_list = []
    h = image.size[0]
    w = image.size[1]
    if len(image.tobytes()) == h*w: is_gray = True
    else: is_gray = False
    if is_gray == False: image = ImageOps.grayscale(image)
    slice_size[0] = min(h, slice_size[0])
    slice_size[1] = min(w, slice_size[1])
    hlist = get_cord(h, max_skip[0], slice_size[0])
    wlist = get_cord(w, max_skip[1], slice_size[1])
    #print(hlist, wlist)
    for sh in hlist:
        for sw in wlist:
            cropped_image = image.crop((sh, sw, sh+slice_size[0], sw+slice_size[1]))
            new_image_list.append(image_to_np(cropped_image))
    new_image_list = np.array(new_image_list).astype(np.float32)
    return new_image_list

def make_batch_with_path(image_path, max_skip, slice_size):
    image = Image.open(image_path)
    return split_image(image, max_skip=max_skip, slice_size=slice_size)

def make_batch_with_np(image_np, max_skip, slice_size):
    print(image_np.shape)
    img = np_to_image(image_np)
    return split_image(img, max_skip, slice_size)

def get_mse(x, y):
    res = np.reshape(x, [-1])
    ans = np.reshape(y, [-1])
    delta = res-ans
    mse = (delta**2).mean(axis=None)
    return mse

def get_psnr(x, y):
    mse = get_mse(x, y)
    return -10*(math.log(mse)/math.log(10))

def get_ssim(x, y):
    x = np.reshape(x, (x.shape[0], x.shape[1]))
    y = np.reshape(y, (y.shape[0], y.shape[1]))
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    xy = np.stack((x,y), axis=0)
    print(x.shape)
    return ssim(x, y, data_range=1., gaussian_weights = True, sigma=1.5, use_sample_convariance = False)
