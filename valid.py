#coding=utf-8 
#加載必要的庫 
import numpy as np 
import cv2
import sys,os 

import glob
#設置當前目錄 
caffe_root = './caffe/'   
sys.path.insert(0, caffe_root + 'python')  
import caffe 
os.chdir(caffe_root)  

caffe.set_mode_gpu()
caffe.set_device(0)

net_file='/home/viplab/桌面/MVCNN_Caffe_40views-master/mvcnn_40view_deploy.prototxt'  
caffe_model='/home/viplab/桌面/MVCNN_Caffe_40views-master/mvcnn_vipModel_40view_0616_iter_10000.caffemodel'  
#mean_file='./ilsvrc_2012_mean.npy' 

net = caffe.Net(net_file,caffe_model,caffe.TEST)  
mean_file = np.load('/home/viplab/桌面/0804_40views/MVCNN_Caffe_40views-master/ilsvrc_2012_mean.npy')
meanPixel = mean_file.mean(1).mean(1)
viewSize = 40
imh = 224
imw = 224
imc = 3

image_dir = r"/home/viplab/桌面/MVCNN_Caffe_40views-master/valid_my" 
file_glob = os.path.join(image_dir,"*")
file_list = []
file_list.extend(glob.glob(file_glob))
#print(file_list)
ims = np.zeros((imh, imw, imc, viewSize ), dtype=np.float32)
for view in range(viewSize):
	image = file_list[view]
	im = cv2.imread(image)
	assert(im is not None)
	im = im.astype(np.float32, copy=False) - meanPixel  
	ims[:,:,:,view] = im
	#ims[:,:,:,view + viewSize] = im
ims = ims.transpose(3,2,0,1)
net.blobs['data'].data[...] = ims
out = net.forward()

imagenet_labels_filename = '/home/viplab/桌面/0804_40views/MVCNN_Caffe_40views-master/list.txt' 
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')  

top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1] 
p= net.blobs['prob'].data
print p
print type(p)
print p.shape
for i in np.arange(top_k.size):  
    print labels[top_k[i]],p[0][top_k[i]] 

