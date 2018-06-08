import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os
import zipfile
from ImageFunctions import *
from PIL import Image as PILImage
from PIL import ImageTk
from skimage.measure import compare_ssim as ssim

#skip_pixel = [40, 40]
dsz = 32
slice_size = [256, 256]
max_skip = [slice_size[0]-dsz, slice_size[1]-dsz]
isExtract = False

var_list = []

# Dummy
c = 1
img_input = tf.placeholder(tf.float32, [None, None, None, c])
img_ans = tf.placeholder(tf.float32, [None, None, None, c])
img_input_cor = tf.placeholder(tf.float32, [None, None, None, c])
img_input_gen = tf.placeholder(tf.float32, [None, None, None, c])
h = tf.placeholder(tf.int32)
w = tf.placeholder(tf.int32)
sess = tf.Session()

def get_data_from_image(path):
    global Data, slice_size, max_skip, img_size, n_prob
    img_size = get_size(path)
    min_sz = min(img_size[0], img_size[1])
    slice_size = [256, 256]
    if min_sz<slice_size[0]:
        slice_size = [min_sz, min_sz]
        max_skip = [slice_size[0]-dsz, slice_size[1]-dsz]
    Data = make_batch_with_path(path, max_skip = max_skip, slice_size = slice_size)
    print(Data.shape)
    n_prob = (img_size[0]*img_size[1])//400

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
    h, w = slice_size[0], slice_size[1]
    img_decom = np.zeros([sz_img, h*w*c], dtype=np.float32)
    for i in range(sz_img):
        for j in range(n_prob):
            inc,val = img_inc[i][j], img_val[i][j]
            img_decom[i][inc] = val
    img_decom = np.reshape(img_decom, [-1, h, w, c])
    return img_decom

img_cor = cor_net(img_input_cor,var_cor['weights'],var_cor['biases'])
img_real_final = img_final + img_cor

saver = tf.train.Saver(var_list)

temp_path = './temp_aic'

def compress(sess, batch_x, new_path, qf):
    feed_dict = {img_input: batch_x, h:slice_size[0], w:slice_size[1]}
    print(img_input.get_shape().as_list(), batch_x.shape)
    img_compressed = sess.run(img_com, feed_dict = feed_dict)
    img_compressed = merge_image(img_compressed, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
    Img_com = np_to_image(img_compressed)
    Img_com.save('img_com.jpeg', quality=qf)
    Img_com.save('img_com2.jpeg', quality=qf)

    if isExtract:
        batch_jpeg = get_jpeg_from_image(img_compressed, qf)
        batch_jpeg = make_batch_with_np(batch_jpeg, max_skip, slice_size)
        feed_dict = {img_input_gen: batch_jpeg, h:slice_size[0], w:slice_size[1]}

        img_fin = sess.run(img_final, feed_dict = feed_dict)
        img_inc, img_val = cor_compress(img_fin, batch_x)

        np.save('img_inc.npy', img_inc)
        np.save('img_val.npy', img_val)

    with zipfile.ZipFile(new_path, 'w') as myzip:
        myzip.write('img_com.jpeg')
        os.remove('img_com.jpeg')
        if isExtract:
            myzip.write('img_inc.npy')
            myzip.write('img_val.npy')
            os.remove('img_inc.npy')
            os.remove('img_val.npy')
    print("COMPRESSED")

def decompress(sess, new_path):
    Img_com = PILImage.open(temp_path+'/img_com.jpeg')
    img_compressed = image_to_np(Img_com)
    os.remove(temp_path+'/img_com.jpeg')

    batch_jpeg = make_batch_with_np(img_compressed, max_skip, slice_size)
    print('Start decompressing')
    feed_dict = {img_input_gen: batch_jpeg, h:slice_size[0], w:slice_size[1]}
    img_fin = sess.run(img_final, feed_dict = feed_dict)

    if os.path.isfile(temp_path+'/img_inc.npy'): isExtract = True
    else: isExtract = False

    if isExtract:
        img_inc = np.load(temp_path+'/img_inc.npy')
        img_val = np.load(temp_path+'/img_val.npy')
        os.remove(temp_path+'/img_inc.npy')
        os.remove(temp_path+'/img_val.npy')
        img_cor_d = cor_decompress(img_inc, img_val)

        feed_dict = {img_input_gen: batch_jpeg, img_input_cor: img_cor_d, h:slice_size[0], w:slice_size[1]}

        img_real_fin = sess.run(img_real_final, feed_dict = feed_dict)
        img_real_fin = merge_image(img_real_fin, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
        img_real_fin = bound_img(img_real_fin)

        Img_fin = np_to_image(img_real_fin)
        if new_path != 'None': Img_fin.save(new_path)
        os.rmdir(temp_path)
        img_fin = merge_image(img_fin, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
        return img_fin, img_real_fin
    else:
        print('End decompressing')
        img_fin = merge_image(img_fin, img_size=img_size, max_skip=max_skip, slice_size=slice_size)
        img_fin = bound_img(img_fin)
        Img_fin = np_to_image(img_fin)
        if new_path != 'None': Img_fin.save(new_path)
        os.rmdir(temp_path)
        return img_fin

def get_res(sess, path, batch_x, qf):
    compress(sess, batch_x, new_path = 'X.aic', qf=qf)
    myzip = zipfile.ZipFile('X.aic', 'r')
    temp_path = './temp_aic'
    myzip.extractall(temp_path)
    if isExtract:
        img_fin, img_real_fin = decompress(sess, new_path = 'None')
        img_real_fin = bound_img(img_real_fin)
        Img_fin = np_to_image(img_real_fin)
    else:
        img_fin = decompress(sess, new_path = 'None')
        img_fin = bound_img(img_fin)
        Img_fin = np_to_image(img_fin)

    ans = make_data_with_path(path)
    ans = bound_img(ans)
    Img_ans = np_to_image(ans)

    Img_ans.thumbnail([200, 200], PILImage.ANTIALIAS)
    Img_fin.thumbnail([200, 200], PILImage.ANTIALIAS)
    return Img_ans, Img_fin

from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter.font as font
import os
import sys

path = ' '
new_path = ' '

def f_path():
    global path, sess
    path = askopenfilename()
    get_data_from_image(path)
    t_path['text'] = path
    print("path : " + path)
    sess.close()
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(device_count={'GPU':0})
    sess = tf.Session(config=config)
    sess.run(init)
    #saver = tf.train.Saver()
    saver.restore(sess, "./model/r-model.ckpt-qf=10-best")

def f_newpath():
    global new_path
    new_path = asksaveasfilename()
    t_newpath['text'] = new_path
    print("new_path : "+ new_path)

def e_ext():
    global isExtract
    isExtract = checkVar.get()

def f_preview():
    print("Started Preview")
    qf = int(t_qf.get())
    print(isExtract, n_prob)
    Img_ans, Img_res = get_res(sess, path = path, batch_x = Data, qf = qf)
    Img_ans, Img_res = ImageTk.PhotoImage(Img_ans), ImageTk.PhotoImage(Img_res)
    photo_ans.configure(image = Img_ans)
    photo_ans.image = Img_ans
    photo_res.configure(image = Img_res)
    photo_res.image = Img_res

def f_start():
    qf = int(t_qf.get())
    print("Started compressing " + path)
    print(Data.shape)
    compress(sess, batch_x = Data, new_path = new_path, qf = qf)
    print("Executed :\n" + S)

def f_cancel():
    print("Cancelled")
    sys.exit()

def f_qf():
    print("QF")
    t_cr['state']='disabled'
    t_qf['state']='normal'

def f_cr():
    print("CR")
    t_cr['state']='normal'
    t_qf['state']='disabled'

root = Tk()
root.title("AIC-compressor")
root.geometry('500x500')
#root.withdraw()

bg = Label(root, text = ' ', anchor=CENTER, bg='white')
bg.place(x=25, y=25, width=450, height=445)

dy = 5
t1 = Label(root, text = 'Image to compress', anchor = W, justify = LEFT, bg='white')
t1.place(x=35, y=35+dy)
t_path = Label(root, text = '', anchor = W, justify = LEFT, bg='white', borderwidth=1, padx=2, pady=2, relief='solid')
t_path.place(x=35, y=60+dy, width=420)
b1 = Button(root, text = "Browse...", anchor = W, justify = RIGHT, padx = 5, command = f_path, borderwidth=1)
b1.place(x=390, y=33+dy)

dy = 60
t2 = Label(root, text = 'Destination', anchor = W, justify = LEFT, bg='white')
t2.place(x=35, y=35+dy)
t_newpath = Label(root, text = '', anchor = W, justify = LEFT, bg='white', borderwidth=1, padx=2, pady=2, relief='solid')
t_newpath.place(x=35, y=60+dy, width=420)
b2 = Button(root, text = "Browse...", anchor = W, justify = RIGHT, padx = 5, command = f_newpath, borderwidth=1)
b2.place(x=390, y=33+dy)

radioVar = IntVar()
radioVar.set(1)
dy = 115
tqf = Radiobutton(root, text = 'Quality Factor (1 ~ 100)', bg='white', command=f_qf, variable=radioVar, value=1)
tqf.place(x=35, y=35+dy)
t_qf = Entry(root, justify = LEFT, bg='white', borderwidth=1,  relief='solid')

t_qf.place(x=35, y=60+dy, width=200)

tcr = Radiobutton(root, text = 'Compression Rate (0 ~ 1)', bg='white', command=f_cr, variable=radioVar, value=2)
tcr.place(x=255, y=35+dy)
t_cr = Entry(root, justify = LEFT, bg='white', borderwidth=1,  relief='solid', state='disabled')
t_cr.place(x=255, y=60+dy, width=200)

checkVar = BooleanVar()
tex = Checkbutton(root, text = 'Additional extract', bg='white', var=checkVar, onvalue=True, offvalue=False, command = e_ext)
tex.place(x=35, y=210)

bPreview = Button(root, text = "Preview", anchor = CENTER, justify = CENTER, padx = 10, command = f_preview)
bPreview.place(x=210, y=210, width=65)

bStart = Button(root, text = "Start", anchor = CENTER, justify = CENTER, padx = 10, command = f_start)
bStart.place(x=300, y=210, width=65)

bStart = Button(root, text = "Cancel", anchor = CENTER, justify = CENTER, padx = 10, command = f_cancel)
bStart.place(x=390, y=210, width=65)

photo_ans = Label()
photo_ans.place(x=35, y=250, width=200, height=200)

photo_res = Label()
photo_res.place(x=255, y=250, width=200, height=200)

root.mainloop()
