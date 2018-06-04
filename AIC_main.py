import argparse
import sys

parser = argparse.ArgumentParser(description='Compress and decompress the image.')
parser.add_argument('--mode', type=str, help = 'mode of the program.\nCompress : com, Decompress : dec, All : all', default = 'all')
parser.add_argument('--path', type = str, help='path of a image or aic file')
parser.add_argument('--new_path', type = str, help='new path of a image', default='result.png')
parser.add_argument('--qf', type=int, help='quality factor of jpeg', default=10)
parser.add_argument('--extract', type=bool, help='decide whether to use additional extract.\nIt consumes extra space, but slightly improves performance by smoothing outliers.', default=False)
args = parser.parse_args()

if args.mode=='all':
    args.extract = True
if args.path==None:
    print("No path to file")
    sys.exit()

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os
import zipfile
from ImageFunctions import *
from PIL import Image
from skimage.measure import compare_ssim as ssim

#skip_pixel = [40, 40]
dsz = 32
slice_size = [256, 256]
max_skip = [slice_size[0]-dsz, slice_size[1]-dsz]

if args.mode=='com' or args.mode=='all':
    img_size = get_size(args.path)
    min_sz = min(img_size[0], img_size[1])
    if min_sz<slice_size[0]:
        slice_size = [min_sz, min_sz]
        max_skip = [slice_size[0]-dsz, slice_size[1]-dsz]
    Data = make_batch_with_path(args.path, max_skip = max_skip, slice_size = slice_size)

    test_size = Data.shape[0]
    print(Data.shape)
    h, w, c = Data.shape[1], Data.shape[2], 1

else:
    myzip = zipfile.ZipFile(args.path, 'r')
    temp_path = './temp_aic'
    myzip.extractall(temp_path)

    img_path = temp_path+"/img_com.jpeg"
    img_size = get_size(img_path)
    min_sz = min(img_size[0], img_size[1])
    if min_sz<slice_size[0]:
        slice_size = [min_sz, min_sz]
        max_skip = [slice_size[0]-dsz, slice_size[1]-dsz]
    Data = make_batch_with_path(img_path, max_skip = max_skip, slice_size = slice_size)
    test_size = Data.shape[0]
    print(Data.shape)
    h, w, c = Data.shape[1], Data.shape[2], 1


n_prob = (img_size[0]*img_size[1])//400
h_prob, w_prob = 3, 3
compress_rate = 4
var_list = []

gpu_device = ['/device:GPU:0', '/device:GPU:1']
img_input = tf.placeholder(tf.float32, [None, h, w, c])
img_ans = tf.placeholder(tf.float32, [None, h, w, c])
img_input_cor = tf.placeholder(tf.float32, [None, h, w, c])
img_input_gen = tf.placeholder(tf.float32, [None, h, w, c])

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(x, alpha * x)

def conv2d(x, W, b, stride = 1, act_func = 'ReLU', use_bn = True, padding = 'SAME'):
    strides = [1, stride, stride, 1]
    sz = 1
    x = tf.nn.conv2d(x, W, strides, padding='SAME')
    x = tf.nn.bias_add(x, b)
    if act_func == 'LReLU': x = leaky_relu(x)
    if act_func == 'ReLU': x = tf.nn.relu(x)
    if act_func == 'TanH': x = tf.nn.tanh(x)
    if act_func == 'Sigmoid': x = tf.nn.sigmoid(x)
    if act_func == 'Softmax': x = tf.nn.softmax(x)
    if act_func == 'None': pass
    if use_bn: return slim.batch_norm(x, fused=False)
    else: return x

def com_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    for i in range(2, 2):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b], act_func = 'LReLU')
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='Sigmoid', use_bn = False)
    return res

def gen_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    for i in range(1, 8//2):
       	x_input = x
        name_w = 'wc'+str(2*i)
        name_b = 'bc'+str(2*i)
        x_input = conv2d(x_input, weights[name_w], biases[name_b], act_func='LReLU')
        name_w = 'wc'+str(2*i+1)
        name_b = 'bc'+str(2*i+1)
        x_input = conv2d(x_input, weights[name_w], biases[name_b], act_func='None')
        x = leaky_relu(x_input + x)
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='TanH', use_bn = False)
    return res

def cor_net(x, weights, biases):
    x = tf.reshape(x, (-1, h, w, c))
    #eturn x
    x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
    for i in range(2, 2):
    	name_w = 'wc'+str(i)
    	name_b = 'bc'+str(i)
    	x = conv2d(x, weights[name_w], biases[name_b], act_func = 'LReLU')
    res = conv2d(x, weights['wcx'], biases['bcx'], act_func='TanH', use_bn = False)
    return res

def make_grad(s, e, name):
    var = tf.get_variable(name, [3, 3, s, e], initializer=tf.contrib.layers.xavier_initializer())
    var_list.append(var)
    return var

def make_bias(x, name):
    var = tf.get_variable(name, [x], initializer=tf.contrib.layers.xavier_initializer())
    var_list.append(var)
    return var

def make_dict(num_layer, num_filter, end_str, s_filter = 1, e_filter = 1):

    result = {}
    weights = {}
    biases = {}

    weights['wc1'] = make_grad(s_filter,num_filter,"w1"+end_str)
    for i in range(2, num_layer):
    	index = 'wc' + str(i)
    	name = 'w' + str(i) + end_str
    	weights[index] = make_grad(num_filter, num_filter, name)
    weights['wcx'] = make_grad(num_filter,e_filter,"wx"+end_str)

    biases['bc1'] = make_bias(num_filter,"b1"+end_str)
    for i in range(2, num_layer):
    	index = 'bc' + str(i)
    	#print(index)
    	name = 'b' + str(i) + end_str
    	biases[index] = make_bias(num_filter, name)
    biases['bcx'] = make_bias(e_filter,"bx"+end_str)

    result['weights'] = weights
    result['biases'] = biases
    return result

var_com = make_dict(5, 32, 'c', 1, 1)
print(var_com)
var_gen = make_dict(18, 32, 'g', 1, 1)
var_img = make_dict(18, 32, 'i', 1, 1)
var_cor = make_dict(18, 32, 'r', 1, 1)

# Define graph

# Com
img_com = com_net(img_input, var_com['weights'], var_com['biases'])
img_res = gen_net(img_com, var_gen['weights'], var_gen['biases'])
img_res_gen = gen_net(img_input_gen, var_gen['weights'], var_gen['biases'])
img_final = img_res_gen + img_input_gen

# Cor
def cor_compress(img, img_ans):
    sz_prob = img.shape[0]
    img_inc = np.zeros([sz_prob, n_prob], dtype=np.uint16)
    img_val = np.zeros([sz_prob, n_prob], dtype=np.float16)
    p_img = np.absolute(img - img_ans)
    for i in range(sz_prob):
        p_img_flatten = np.reshape(p_img[i], [-1])
        img_flatten = np.reshape(img_ans[i] - img[i], [-1])
        incides = np.argpartition(p_img_flatten,-n_prob)[-n_prob:]
        for j in range(n_prob):
            inc = incides[j]
            img_inc[i][j] = inc
            img_val[i][j] = img_flatten[inc]
    return img_inc, img_val

def cor_decompress(img_inc, img_val):
    sz_img = img_inc.shape[0]
    img_decom = np.zeros([sz_img, h*w*c], dtype=np.float32)
    for i in range(sz_img):
        for j in range(n_prob):
            inc,val = img_inc[i][j], img_val[i][j]
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, c])
    return img_decom

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])
img_real_final = img_final + img_cor

# Initializing the variables
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
max_acc = 0.
qf = args.qf
p_acc = []

saver = tf.train.Saver(var_list)

def compress(sess, batch_x, new_path):
    feed_dict = {img_input: batch_x}
    img_compressed = sess.run(img_com, feed_dict = feed_dict)
    img_compressed = merge_image(img_compressed, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
    Img_com = np_to_image(img_compressed)
    Img_com.save('img_com.jpeg', quality=qf)

    if args.extract:
        batch_jpeg = get_jpeg_from_image(img_compressed, qf)
        batch_jpeg = make_batch_with_np(batch_jpeg, max_skip, slice_size)
        feed_dict = {img_input_gen: batch_jpeg}

        img_fin = sess.run(img_final, feed_dict = feed_dict)
        img_inc, img_val = cor_compress(img_fin, batch_x)

        np.save('img_inc.npy', img_inc)
        np.save('img_val.npy', img_val)

    with zipfile.ZipFile(new_path, 'w') as myzip:
        myzip.write('img_com.jpeg')
        os.remove('img_com.jpeg')
        if args.extract:
            myzip.write('img_inc.npy')
            myzip.write('img_val.npy')
            os.remove('img_inc.npy')
            os.remove('img_val.npy')


def decompress(sess, new_path):
    Img_com = Image.open(temp_path+'/img_com.jpeg')
    img_compressed = image_to_np(Img_com)
    os.remove(temp_path+'/img_com.jpeg')

    batch_jpeg = get_jpeg_from_image(img_compressed, qf)
    batch_jpeg = make_batch_with_np(batch_jpeg, max_skip, slice_size)
    print('Start decompressing')
    feed_dict = {img_input_gen: batch_jpeg}
    img_fin = sess.run(img_final, feed_dict = feed_dict)

    if os.path.isfile(temp_path+'/img_inc.npy'): isExtract = True
    else: isExtract = False

    if isExtract:
        img_inc = np.load(temp_path+'/img_inc.npy')
        img_val = np.load(temp_path+'/img_val.npy')
        os.remove(temp_path+'/img_inc.npy')
        os.remove(temp_path+'/img_val.npy')
        img_cor_d = cor_decompress(img_inc, img_val)

        feed_dict = {img_input_gen: batch_jpeg, img_input_cor: img_cor_d}

        img_real_fin = sess.run(img_real_final, feed_dict = feed_dict)
        img_real_fin = merge_image(img_real_fin, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
        img_real_fin = bound_img(img_real_fin)

        Img_fin = np_to_image(img_real_fin)
        Img_fin.save(new_path)
        os.rmdir(temp_path)
        img_fin = merge_image(img_fin, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
        return img_fin, img_real_fin
    else:
        print('End decompressing')
        img_fin = merge_image(img_fin, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
        img_fin = bound_img(img_fin)
        Img_fin = np_to_image(img_fin)
        Img_fin.save(new_path)
        os.rmdir(temp_path)
        return img_fin

with sess:
    sess.run(init)
    #saver = tf.train.Saver()
    saver.restore(sess, "./model/r-model.ckpt-qf=10-best")
    step = 0
    tot_mse = 0.
    if args.mode == 'com':
        compress(sess, Data, new_path = args.new_path)
    elif args.mode == 'dec':
    	decompress(sess, new_path = args.new_path)
    elif args.mode == 'all':
        compress(sess, Data, new_path = 'X.aic')
        myzip = zipfile.ZipFile('X.aic', 'r')
        temp_path = './temp_aic'
        myzip.extractall(temp_path)
        img_fin, img_real_fin = decompress(sess, new_path = 'X.png')
        img_fin = bound_img(img_fin)
        img_real_fin = bound_img(img_real_fin)

        ans = make_data_with_path(args.path)

        batch_jpeg = get_jpeg_from_image(ans, qf)
        batch_jpeg = bound_img(batch_jpeg)

        Img_jpeg = np_to_image(batch_jpeg)
        Img_jpeg.save('result_jpeg.jpeg', qf=qf)
        print(str(step)+": "+"PSNR = {:.5f}".format(get_psnr(batch_jpeg, ans))+" SSIM = {:.5f}".format(get_ssim(batch_jpeg, ans)))

        #rint(gloss, get_mse(img_fin, ans))
        print(str(step)+": "+"PSNR = {:.5f}".format(get_psnr(img_fin, ans))+" SSIM = {:.5f}".format(get_ssim(img_fin, ans)))
        #rint(rloss, get_mse(img_real_fin, ans))
        print(str(step)+": "+"PSNR = {:.5f}".format(get_psnr(img_real_fin, ans))+" SSIM = {:.5f}".format(get_ssim(img_real_fin, ans)))
    else:
        print("Invalid mode")
        sys.exit()
    #print(tot_mse/test_size)
print('FINISHED!!!!!')
