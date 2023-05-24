'''
Author: Dr Yasmeen George
Email: yasmeen.mourice@gmail.com
Date: 24 May 2023
'''

import numpy as np
from PIL import Image
import skfuzzy as fuzz
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage import color
from skimage.segmentation import watershed
from skimage import filters
from scipy import ndimage
from skimage import morphology
from skimage.morphology import square
from skimage.segmentation import mark_boundaries
from skimage import exposure

def get_image_features(imfile,features_equalized_flag):
    image = iio.imread(imfile)
    if features_equalized_flag:
        R_adapteq = exposure.equalize_adapthist(image[:, :, 0], clip_limit=0.03)
        G_adapteq = exposure.equalize_adapthist(image[:, :, 1], clip_limit=0.03)
        B_adapteq = exposure.equalize_adapthist(image[:, :, 2], clip_limit=0.03)
        tmp = (np.dstack((R_adapteq, G_adapteq, B_adapteq)) * 255).astype(np.uint8)
    else:
        tmp = image
    I = Image.fromarray(tmp)

    ycbcr = np.asarray(I.convert('YCbCr'))
    hsv = np.asarray(I.convert('HSV'))

    sig_filt = 4
    smooth_I  = filters.gaussian(tmp, sigma=sig_filt,channel_axis=2)
    smooth_I = Image.fromarray(np.asarray(smooth_I*255,np.uint8))

    lab = color.rgb2lab(smooth_I)
    rows= image.shape[0]
    cols = image.shape[1]
    features = np.zeros((rows*cols , 12), np.float32)
    indices = np.zeros((rows*cols , 2), np.int32)
    index =0

    for i in range(rows):
        for j in range(cols):
            if True: #image_mask[i,j] != 0:
                features[index,0:3]= tmp[i,j,:]
                features[index,3:6]= ycbcr[i,j,:]
                features[index,6:9]= hsv[i,j,:]
                features[index,9:12]=lab[i,j,:]
                indices[index,0]=i
                indices[index,1]=j

                index +=1

    features = features[:index, ...]
    indices = indices[:index, ...]
    return image, features,indices


def otsu_segmentation(I,features,indices):

    levels = ['R', 'G','B', 'Y', 'Cb', 'Cr', 'H', 'S', 'V', 'L' ,'a*' ,'b*']

    L_INDICES = [10]
    for level_index in L_INDICES:
        pix_vec = features[:,level_index]
        channel_name  = levels[level_index]

        val = filters.threshold_otsu(pix_vec)
        if level_index < 4 :
            thresh = pix_vec < val
        else:
            thresh = pix_vec > val
        foreground = np.zeros(I.shape,np.uint8)
        background = np.zeros(I.shape,np.uint8)
        skin_mask = np.zeros((I.shape[0],I.shape[1]),np.float32)

        for i in range(indices.shape[0]):
            if thresh[i] == 1:
                foreground[indices[i,0],indices[i,1],:] = features[i,0:3]
            if thresh[i] == 0:
                background[indices[i,0],indices[i,1],:] = features[i,0:3]
            skin_mask[indices[i,0],indices[i,1]] = thresh[i] * 255 #pix_vec[i]
    return foreground, background, skin_mask , channel_name
def watershed_segmentation(I,features,indices,save_name):
    #img_gray = rgb2gray(I)
    n =6
    m = 5
    fs = 6
    index=0

    levels_inds = [5,6,10]
    levels = ['cr', 'H' , 'a*']
    for k in range(len(levels_inds)):
        pix_vec =features[:,levels_inds[k]]
        L_name  = levels[k]

        val = filters.threshold_otsu(pix_vec)
        thresh = pix_vec > val

        equalized_I = np.zeros(I.shape,np.float32)
        otsu_I = np.zeros((I.shape[0],I.shape[1]),np.uint8)
        gray_I = np.zeros((I.shape[0],I.shape[1]),np.float32)
        mask_I = np.zeros((I.shape[0],I.shape[1]),np.int32)
        for i in range(indices.shape[0]):
            if thresh[i] == 1:
                otsu_I[indices[i,0],indices[i,1]] =thresh[i]
            gray_I[indices[i,0],indices[i,1]] = pix_vec[i]
            mask_I[indices[i,0],indices[i,1]] = 1
            equalized_I[indices[i,0],indices[i,1],:] = features[i,0:3]

        closed = morphology.closing(otsu_I, square(3))
        dilated = morphology.dilation(closed, square(3))
        distance = ndimage.distance_transform_edt(dilated)
        local_maxi1 = distance > distance.max()*.1
        unknown = dilated - local_maxi1 # good  edges image

        markers = morphology.label(local_maxi1)
        markers2 = markers+1

        markers2[unknown==1] = 0
        labels_ws = watershed(gray_I, markers2, mask=mask_I)
        marked = mark_boundaries(I, labels_ws,color=(0,0,1))

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(gray_I,cmap=plt.cm.gray)
        plt.title(L_name,fontsize=fs)
        plt.xticks([])
        plt.yticks([])


        index +=1
        plt.subplot(n,m,index)
        plt.imshow(otsu_I,cmap=plt.cm.gray)
        plt.title('1- Otsu',fontsize=fs)#1- Otsu Threshold on H level
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(closed,cmap=plt.cm.gray)
        plt.title('2- Closing',fontsize=fs)#2- Closing
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(dilated,cmap=plt.cm.gray)
        plt.title('3- dilation',fontsize=fs)#3- dilation
        plt.xticks([])
        plt.yticks([])
   
        index +=1
        plt.subplot(n,m,index)
        plt.imshow(distance,cmap=plt.cm.gray)
        plt.title('4- distance',fontsize=fs)# 4- distance transform
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(local_maxi1,cmap=plt.cm.gray)
        plt.title('5- thres_distance',fontsize=fs)#local maximum points from threshold distance
        plt.xticks([])
        plt.yticks([])


        index +=1
        plt.subplot(n,m,index)
        plt.imshow(unknown,cmap=plt.cm.gray)
        plt.title('subtract',fontsize=fs)#distance thresholded subtraction from dilated image
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(labels_ws,cmap=plt.cm.gray)
        plt.title('6- labels_ws',fontsize=fs)# watershed labels
        plt.xticks([])
        plt.yticks([])


        index +=1
        plt.subplot(n,m,index)
        plt.imshow(equalized_I,cmap=plt.cm.gray)
        plt.imshow(labels_ws, cmap=plt.cm.Spectral, alpha=.4)
        plt.title('7- watershed',fontsize=fs)# segmented image using watershed
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(marked,cmap=plt.cm.gray)
        plt.title('8- segmented boundary',fontsize=fs)# segmented image using watershed
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)

    plt.close('all')

def fuzzy_clustering(I,features,indices,save_name):

    X_old = features[:,5:8]
    X_old[:,2]= features[:,10]

    from sklearn.kernel_approximation import RBFSampler

    rbf_feature = RBFSampler(gamma=0.2, random_state=1)
    X = rbf_feature.fit_transform(X_old)

    n = 2
    m = 3
    fs = 6
    index=0
    X = X.T
    nclusters = 2
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X,2 , 2, error=0.005, maxiter=1000)
    plt.figure()
    cluster_membership = np.argmax(u, axis=0)

    equalized_I = np.zeros(I.shape,np.float32)
    pixels_labels = np.zeros((I.shape[0],I.shape[1]),np.int32)
    for i in range(indices.shape[0]):
        pixels_labels[indices[i,0],indices[i,1]] = cluster_membership[i]+1
        equalized_I[indices[i,0],indices[i,1],:] = features[i,0:3]


    index +=1
    plt.subplot(n,m,index)
    plt.imshow(I)
    plt.title('I',fontsize=fs)# segmented image using watershed
    plt.xticks([])
    plt.yticks([])

    index +=1
    plt.subplot(n,m,index)
    plt.imshow(equalized_I)
    plt.title('I',fontsize=fs)# segmented image using watershed
    plt.xticks([])
    plt.yticks([])

    index +=1
    plt.subplot(n,m,index)
    for j in range(nclusters):
        x= X_old[cluster_membership == j,0]
        y= X_old[cluster_membership == j,1]
        plt.plot(x,y, 'o', label='cluster ' + str(j+1))
    plt.title('clusters',fontsize=fs)
    plt.xticks([])
    plt.yticks([])
    index +=1
    plt.subplot(n,m,index)
    plt.imshow(pixels_labels,cmap=plt.cm.gray)
    plt.title('labels',fontsize=fs)# segmented image using watershed
    plt.xticks([])
    plt.yticks([])



    for k in range(1,nclusters+1):
        res = I *1

        res[pixels_labels != k,:] = 0

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(res)
        plt.title('cluster' + str(k),fontsize=fs)# segmented image using watershed
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(save_name,dpi=300)

    plt.close('all')


def segment_image(im_path,save_path=None):
    with_equalization = 0
    image, features, indices = get_image_features(im_path, with_equalization)

    # Otsu Thresholding
    foreground, background, skin_mask, channel_name = otsu_segmentation(image, features, indices)

    if save_path:

        sv_path = save_path + im_path.split('/')[-1][:-4]

        plt.imsave(sv_path+ "_skin_mask.jpg", skin_mask, cmap=plt.cm.gray)
        plt.imsave(sv_path+ "_skin_image.jpg", foreground, cmap=plt.cm.gray)

        plt.figure(figsize=(20, 20))
        plt.subplot(2,2,1), plt.imshow(image), plt.title('image'), plt.axis('off')
        plt.subplot(2,2,2), plt.imshow(foreground), plt.title('foreground'), plt.axis('off')
        plt.subplot(2, 2, 3), plt.imshow(background), plt.title('background'), plt.axis('off')
        plt.subplot(2, 2, 4), plt.imshow(skin_mask, cmap=plt.cm.gray), plt.title('skin mask'), plt.axis('off')
        plt.savefig(sv_path + '_otsu_demo.png', bbox_inches='tight', dpi=300)
        plt.close('all')

    ### Other comparative methods:
    #fuzzy_clustering(image, features, indices, sv_path + "_fuzzy_clustering_demo")

    #watershed_segmentation(image, features, indices, sv_path + "_watershed_demo")

    return image, skin_mask, foreground
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path', default="../sample_images/img_arm.jpg", type=str)
    parser.add_argument('--save_path', default="../sample_results/skin/", type=str)
    args = parser.parse_args()

    image, skin_mask, foreground = segment_image(args.im_path, args.save_path)
    ## Sample Example
    ## python skin_detection.py --im_path ../sample_images/img_arm.JPG --save_path ../sample_results/skin/