from PIL import Image, ImageOps
import fnmatch
import os
import copy
import numpy as np
import math

def normalize_img(img):
    #img = img.astype(np.float32)
    #return (img-128)/128
    return img/255.

def denormalize_img(img):
    #return ((img+1.)*128).astype(np.uint8)
    #img = np.maximum(img, 0)
    return (img*255).astype(np.uint8)

def merge_image(img, skip_pixel = [-1, -1]):
    H, W, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    if skip_pixel[0]==-1: skip_pixel[0] = h
    if skip_pixel[1]==-1: skip_pixel[1] = w
    new_h, new_w = h+(H-1)*skip_pixel[0], w+(W-1)*skip_pixel[1]
    new_img = np.zeros((new_h, new_w))
    chk_img = np.zeros((new_h, new_w))
    for i in range(H):
        for j in range(W):
            for k in range(h):
                kk = i*skip_pixel[0] + k
                for l in range(w):
                    ll = j*skip_pixel[1] + l
                    new_img[kk][ll]+=img[j][i][k][l]
                    chk_img[kk][ll]+=1
    for i in range(new_h):
        for j in range(new_w):
            new_img[i][j]/=chk_img[i][j]

    print(chk_img)
    return new_img

def get_jpeg_from_batch(img_np, h, w, qf = 10):
    img_np = denormalize_img(img_np)
    n = img_np.shape[0]
    res_img = []
    for i in range(n):
        img = np.reshape(img_np[i], (h, w))
        img = Image.fromarray(img, 'L')
        img.save('temp.jpeg', quality=100)
        img.save('temp2.jpeg', quality=qf)
        img = Image.open('temp2.jpeg')
        img.thumbnail((h, w), Image.ANTIALIAS)
        img = image_to_np(img, h, w)
        res_img.append(img)
    res_img = np.array(res_img)
    #print(res_img.shape)
    return res_img
    
def np_to_image(img_np, size, skip_pixel):
    img = np.reshape(img_np, size)
    img = np.maximum(img, 0.)
    img = np.minimum(img, 1.)
    img = merge_image(img, skip_pixel = skip_pixel)
    img = denormalize_img(img)
    #img = np.reshape(img, (img.shape[1], img.shape[2]))
    img = Image.fromarray(img, 'L')
    return img

def image_to_np(img, h, w):
    image_np = np.fromstring(img.tobytes(), dtype=np.uint8).astype(float)
    print(image_np.shape)
    image_np = image_np.reshape((h, w, 1))
    image_np = normalize_img(image_np)
    #image_np/=255.
    return image_np

def make_data_with_image(image_path, isTest = False):
    #global h, w
    image = Image.open(image_path)
    new_image_list = []
    h = image.size[0]
    w = image.size[1]
    if len(image.tobytes()) == h*w: is_gray = True
    else: is_gray = False
    if is_gray==False: image = ImageOps.grayscale(image)
    
    res = []
    res.append(image_to_np(image, h, w))
    return np.array(res)#$image_to_np(image) 

def make_batch_with_path(image_path, size = (40, 40), skip_pixel = [-1, -1]):
    image = Image.open(image_path)
    return make_batch_with_image(image, size,skip_pixel)

def make_batch_with_np(image_np, size = (40, 40), skip_pixel = [-1, -1]):
    h, w = image_np.shape[1], image_np.shape[2]
    image_np = np.reshape(image_np[0], (h, w))
    img = Image.fromarray(image_np,  'L')
    #img.save('temp3.png')
    return make_batch_with_image(img, size, skip_pixel)

def make_batch_with_image(image, size = (40, 40), skip_pixel = [-1, -1]):
    if skip_pixel[0] == -1: skip_pixel[0] = size[0]
    if skip_pixel[1] == -1: skip_pixel[1] = size[1]
    #image = Image.open(image_path)
    new_image_list = []
    h = image.size[0]
    w = image.size[1]
    if len(image.tobytes()) == h*w: is_gray = True
    else: is_gray = False
    if is_gray==False: image = ImageOps.grayscale(image)
    H, W = ((h-size[0])//skip_pixel[0]) + 1, ((w-size[1])//skip_pixel[1]) + 1
    image.thumbnail((size[0]+(H-1)*skip_pixel[0], size[1]+(W-1)*skip_pixel[1]), Image.ANTIALIAS)
    #image.save('WTF.png')
    for i in range(H):
        sh = i*skip_pixel[0]
        for j in range(W):
            sw = j*skip_pixel[1]
            cropped_image = image.crop((sh, sw, sh+size[0], sw+size[1]))
            #cropped_image.save('WTF.png')
            new_image_list.append(image_to_np(cropped_image, size[0], size[1]))
    new_image_list = np.array(new_image_list).astype(np.float32)
    return new_image_list

def get_mse(x, y):
    res = np.reshape(x, [-1])
    ans = np.reshape(y, [-1])
    delta = res-ans
    mse = (delta**2).mean(axis=None)
    return mse

def get_psnr(x, y):
    mse = get_mse(x, y)
    return -10*(math.log(mse)/math.log(10))

'''
matches = []
cnt = 0
for root, dirnames, filenames in os.walk('./images/final'):
    for filename in fnmatch.filter(filenames, '*'):
        matches.append(os.path.join(root, filename)) 
        cnt+=1

print("%d files found" % cnt)

tot_image = []

for i in range(cnt):
    F = matches[i]
    images = make_data_with_image(F)
    tot_image.append(images)

tot_image = np.array(tot_image)
print(tot_image.shape)
np.save("data_final.npy", tot_image)
'''
