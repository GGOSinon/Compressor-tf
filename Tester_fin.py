import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from ImageFunctions import *
from PIL import Image
from skimage.measure import compare_ssim as ssim
import argparse
import sys

parser = argparse.ArgumentParser(description='Compress and decompress the image.')
parser.add_argument('--mode', type=str, help = 'Mode of the program. Compress : com, Decompress : dec, All : all', default = 'all')
parser.add_argument('--path', type = str, help='path of a image or pkl')
parser.add_argument('--new_path', type = str, help='new path of a image', default='result.png')
parser.add_argument('--qf', type=int, help='quality factor of jpeg', default=10)
args = parser.parse_args()

if args.path==None:
    print("No path to image")
    sys.exit()

#skip_pixel = [40, 40]
h, w, c = 256, 256, 1
skip_pixel = [h, w]
size = [256, 256]

if args.mode=='com' or args.mode=='all':
    Data = make_batch_with_path(args.path, size = [h, w], skip_pixel = skip_pixel)
    #Data = np.load("../data/data_val.npy")
    test_size = Data.shape[0]
    print(Data.shape)
    c, h, w = Data.shape[3], Data.shape[1], Data.shape[2]

n_prob = (size[0]*size[1])//400
h_prob, w_prob = 3, 3
#c, h, w = 1, 256, 256
compress_rate = 4
qf = 10
var_list = []
#hh, ww = h/compress_rate, w/compress_rate

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
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides, padding='SAME')
    #x = tf.pad(x, [[0, 0],[sz, sz], [sz, sz], [0, 0]], "SYMMETRIC")
    #x = tf.nn.conv2d(x, W, strides, padding='VALID')
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
    #fc1 = conv2d(conv3, weights['wd1'], biases['bd1'])
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
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

#learning_rate = start_learning_rate

def next_batch(x, size=-1):
    data = []
    if size==-1:
        data.append(Data[x])
    else:
    	for i in range(size):
    		data.append(Data[x+i])
    data = np.array(data)
    data = data.reshape(-1, h, w, c)
    return data, data

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
    #img_res = np.array(img_res)
    #img_input_cor = tf.convert_to_tensor(img_input_cor_np)
    return img_inc, img_val

def bound_img(img):
    img_fin = img
    img_fin = np.maximum(img_fin, 0.)
    img_fin = np.minimum(img_fin, 1.)
    img_fin = denormalize_img(img_fin)
    img_fin = normalize_img(img_fin)
    return img_fin

def cor_decompress(img_inc, img_val):
    sz_img = img_inc.shape[0]
    img_decom = np.zeros([sz_img, h*w*c], dtype=np.float32)
    for i in range(sz_img):
        for j in range(n_prob):
            inc,val = img_inc[i][j], img_val[i][j]
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, c])
    return img_decom

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
    #x = bound_img(xx)
    #y = bound_img(yy)
    x = np.reshape(x, size)
    y = np.reshape(y, size)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    xy = np.stack((x,y), axis=0)
    print(x.shape)
    return ssim(x, y, data_range=1., gaussian_weights = True, sigma=1.5, use_sample_convariance = False)

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])

img_real_final = img_final + img_cor
# Define loss and optimizer
com_loss = tf.losses.mean_squared_error(img_ans - img_com, img_res)
gen_loss = tf.losses.mean_squared_error(img_ans, img_final)
#img_loss = tf.losses.mean_squared_error(img_ans_prob, img_prob)
#img_loss = tf.losses.softmax_cross_entropy(img_ans_prob, img_prob)
cor_loss = tf.losses.mean_squared_error(img_ans - img_final, img_cor)
# Initializing the variables
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
max_acc = 0.

p_acc = []
#Log = open('Log_cg.txt', 'w')
#Log.close()

saver = tf.train.Saver(var_list)
#saver = tf.train.Saver()

def compress(sess, batch_x, batch_y):
    H, W = 1+((size[0]-h)//skip_pixel[0]), 1+((size[1]-w)//skip_pixel[1])
    size[0], size[1] = (H-1)*skip_pixel[0] + h, (W-1)*skip_pixel[0] + w
    #print(H, W, h, w)
    feed_dict = {img_input: batch_x}
    img_compressed = sess.run(img_com, feed_dict = feed_dict) 
    batch_jpeg = get_jpeg_from_batch(img_compressed, h, w, qf)
    
    feed_dict = {img_input: batch_x, img_input_gen: batch_jpeg}
    
    img_fin = sess.run(img_final, feed_dict = feed_dict)
    img_inc, img_val = cor_compress(img_fin, batch_y)
    
    Img_com = np_to_image(img_compressed, (H, W, h, w), skip_pixel)
    Img_com.save('img_com.jpeg', quality=qf)
    
    np.save('img_inc.npy', img_inc)
    np.save('img_val.npy', img_val)

def decompress(sess, path):
    H, W = 1+((size[0]-h)//skip_pixel[0]), 1+((size[1]-w)//skip_pixel[1])
    size[0], size[1] = (H-1)*skip_pixel[0] + h, (W-1)*skip_pixel[0] + w
    
    Img_com = Image.open(path+'img_com.jpeg')
    img_compressed = np.array([image_to_np(Img_com, size[0], size[1])])
    batch_jpeg = get_jpeg_from_batch(img_compressed, h, w, qf)
    feed_dict = {img_input_gen: batch_jpeg}
    img_fin = sess.run(img_final, feed_dict = feed_dict)
    img_inc = np.load(path+'img_inc.npy')
    img_val = np.load(path+'img_val.npy')
    
    img_cor_d = cor_decompress(img_inc, img_val)

    feed_dict = {img_input_gen: batch_jpeg, img_input_cor: img_cor_d}

    img_real_fin = sess.run(img_real_final, feed_dict = feed_dict)
    
    Img_fin = np_to_image(img_real_fin, (H, W, h, w), skip_pixel)
    Img_fin.save(args.new_path)
    return img_fin, img_real_fin
    
with sess: 
    sess.run(init)
    #saver = tf.train.Saver()
    saver.restore(sess, "./model/best/r-model.ckpt-qf=10-best")
    step = 0
    tot_mse = 0.
    if args.mode == 'com':
        batch_x, batch_y = next_batch(step, test_size)
        compress(sess, batch_x, batch_y)
    elif args.mode == 'dec':
    	decompress(sess, args.path) 
    elif args.mode == 'all':#test_size:
        batch_x, batch_y = next_batch(step, test_size) 
       
        compress(sess, batch_x)
        img_fin, img_real_fin = decompress(sess)
        img_fin = bound_img(img_fin)
        img_real_fin = bound_img(img_real_fin)
        
        ans = batch_y
        #print(img_fin, ans)
        for i in range(test_size):
            tot_mse+=get_mse(img_fin[i], ans[i])
        
        batch_jpeg = get_jpeg_from_batch(batch_x, h, w, qf)
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
