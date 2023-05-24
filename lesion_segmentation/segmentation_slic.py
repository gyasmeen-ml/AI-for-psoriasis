'''
Author: Dr Yasmeen George
Email: yasmeen.mourice@gmail.com
Date: 24 May 2023
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from PIL import Image
import argparse
from skimage.io import imread
from scipy.stats import entropy as ent
import numpy as np
from skimage.measure import shannon_entropy

import os
import time
from skimage import filters

def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig=signal.size
    symset=list(set(signal))

    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def matrix_entropy_rows(matrix):
    entropy_vec = np.zeros(matrix.shape[0], np.float32)

    for i in range(matrix.shape[0]):
        entropy_vec[i] = entropy(matrix[i,:])

    return entropy_vec

def load_test_data(file_name='/home/ygeorge/PsoriasisDatasets/LesionSegmentation/train_noext.txt'):
    file_names = []
    with open(file_name, 'rb') as f:
        for fn in f:
            file_names.append(fn.strip())

    return file_names

def create_full_paths(file_names, image_dir, label_dir, image_ext='.jpg', label_ext='.png'):
  image_paths = []
  label_paths = []

  for file_name in file_names:
      image_paths.append(os.path.join(image_dir, file_name+image_ext))
      label_paths.append(label_dir +file_name+label_ext)
  return image_paths, label_paths
def get_segments_features(segments , image, gt, smooth_flag=False,sig_filt=10):


    if smooth_flag == True:
        image = filters.gaussian(image, sigma=sig_filt,multichannel=True)
        #props = regionprops(segments)
    gt = gt.astype(np.float32)/255

    num_segments = segments.max() + 1
    num_features = 5 # min,max,avg,var,entropy
    num_channels = 12 # R G B Y Cb Cr H S V L A B

    features_mat = np.zeros((num_segments,num_channels*num_features),np.float32)
    labels = np.zeros(num_segments,np.float32)


    sample_segment = np.zeros(segments.shape).astype(bool)

    I=Image.fromarray(np.asarray(image*255 , np.uint8))

    ycbcr = img_as_float(np.asarray(I.convert('YCbCr')))
    hsv = img_as_float(np.asarray(I.convert('HSV')))
    from skimage import color
    lab = color.rgb2lab(I)



    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    y=ycbcr[:,:,0]
    cb = ycbcr[:,:,1]
    cr = ycbcr[:,:,2]
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v= hsv[:,:,2]
    l = lab[:,:,0]
    a= lab[:,:,1]
    b=lab[:,:,2]


    for i in range(num_segments):
        sample_segment[:,:] = 0
        tmp = np.where(segments == i)
        y_ind = tmp[0]
        x_ind = tmp[1]

        sample_segment[y_ind,x_ind]=1
        if np.sum(sample_segment) == 0:
            continue
        ''' 
        plt.imsave(str(i)+'.png', sample_segment,cmap = plt.cm.gray)	
        sample_segment[:,:] = 0
        '''

        r_vals = R[sample_segment]
        g_vals = G[sample_segment]
        b_vals = B[sample_segment]

        segment_pixels = [r_vals]
        segment_pixels = np.append(segment_pixels, [g_vals],axis=0)
        segment_pixels = np.append(segment_pixels, [b_vals],axis=0)

        #print (segment_pixels.shape , np.min(segment_pixels,axis=1) , np.max(segment_pixels,axis=1))
        #print (len(r_vals) , r_vals.min() , r_vals.max())
        #print g_vals.shape , g_vals.min() , g_vals.max()
        #print b_vals.shape , b_vals.min() , b_vals.max()

        segment_pixels = np.append(segment_pixels, [y[sample_segment]],axis=0)
        segment_pixels = np.append(segment_pixels, [cb[sample_segment]],axis=0)
        segment_pixels = np.append(segment_pixels, [cr[sample_segment]],axis=0)


        segment_pixels = np.append(segment_pixels, [h[sample_segment]],axis=0)
        segment_pixels = np.append(segment_pixels, [s[sample_segment]],axis=0)
        segment_pixels = np.append(segment_pixels, [v[sample_segment]],axis=0)

        segment_pixels = np.append(segment_pixels, [l[sample_segment]],axis=0)
        segment_pixels = np.append(segment_pixels, [a[sample_segment]],axis=0)
        segment_pixels = np.append(segment_pixels, [b[sample_segment]],axis=0)
        #print segment_pixels.shape , np.min(segment_pixels,axis=1) , np.max(segment_pixels,axis=1)
        # segment_pixels 12 x n (n is the num pixel in segment)
        segment_pixels = np.asarray(segment_pixels)
        tmp = np.asarray([np.min(segment_pixels,axis=1), np.max(segment_pixels,axis=1), np.mean(segment_pixels,axis=1) ,np.var(segment_pixels,axis=1)]).flatten()
        #print(i, segment_pixels.shape, tmp.shape)
        ent_vec = matrix_entropy_rows(segment_pixels)

        features_mat[i,:]  = np.append(tmp, ent_vec)



        labels[i]= np.mean(gt[sample_segment])

        sample_segment[:,:] = 0
    return features_mat,labels

def main(wr):
    file_names = load_test_data()

    images, labels = create_full_paths(file_names, '/home/ygeorge/PsoriasisDatasets/LesionSegmentation/All_Images/', '/home/ygeorge/PsoriasisDatasets/LesionSegmentation/All_Images/BW_')
    index = -1
    for img_path, label_path in zip(images, labels):
        print ('Processing image #: ', index+2 , ' out of ' , len(images))
        gt = imread(label_path)
        img = imread(img_path)

        image = img_as_float(img)
        index = index+1

        superpixels  = [100]#[100,200,300]

        for numSegments in superpixels:
            # apply SLIC and extract (approximately) the supplied number
            # of segments
            segments = slic(image, n_segments = numSegments, sigma = 5)
            #np.save(SavePath + '/segments/'+ file_names[index] , segments)
            features_mat,labels = get_segments_features(segments , image, gt)

        if index == 0 : # write HEADER
            header  = ['IMG_ID']
            for i in range(features_mat.shape[1]): # 60 features
                header.append('feat'+str(i+1))
                header.append('segment_lbl_prob')
                wr1.writerow(header)

        for i in range(features_mat.shape[0]):
            f = features_mat[i,:]
            curr_row = np.append([index], f)
            curr_row = np.append(curr_row, [labels[i]])
            wr.writerow(curr_row)

        del  features_mat , labels ,segments

        ''' ############# trace segments are correct or not ############# 
        # show the output of SLIC
        ax = fig.add_subplot(1, subfigs_num, numSegments/100)
        ax.imshow(mark_boundaries(image, segments))
        ax.set_title("Superpixels -- %d segments" % (numSegments), fontsize=8)
        plt.axis("off")
        print (index, segments.min() , segments.max() ,segments.dtype )
        '''
        # show the plots
        #plt.savefig(SavePath +file_names[index] +'.png',dpi=300)
        #plt.close("all")
        #fig.clf()

        #plt.show()

#import csv
#myfile1 = open('train_superpixel_features1.csv', 'wb')
#wr1 = csv.writer(myfile1, quoting=csv.QUOTE_ALL)
#main(wr1)
#myfile1.close()

