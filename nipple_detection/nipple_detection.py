'''
Author: Dr Yasmeen George
Email: yasmeen.mourice@gmail.com
Date: 24 May 2023
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage import filters
import numpy as np
from scipy import ndimage
import imageio.v3 as iio
from PIL import Image
from skimage.measure import label
from skimage.measure import regionprops
from skimage import color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage import morphology
from skimage.draw import circle_perimeter
from skimage.draw import disk as circle
from skimage.metrics import structural_similarity as ssim
import argparse
import sys
sys.path.insert(1, '../')
from skin_detection.skin_detection import segment_image



def get_resized_image(raw_image , image, mask):
    mask[mask < 100] = 0

    r, c = np.where(mask > 0)
    rmin = np.amin(r)
    rmax = np.amax(r)
    cmin = np.amin(c)
    cmax = np.amax(c)

    mask = mask[rmin:rmax, cmin:cmax]
    image = image[rmin:rmax, cmin:cmax]

    image[mask == 0, :] = 0

    d_h = 512

    if image.shape[0] < image.shape[1]:  # h is smaller than w
        image = image.transpose((1, 0, 2))
        image_mask = mask.transpose((1, 0))
    # print 'image after transpose' , image.shape

    new_h = d_h
    ratio = d_h / float(image.shape[0])
    new_w = int((float(image.shape[1]) * float(ratio)))

    image = np.array(Image.fromarray(image).resize((new_w, new_h)))
    mask = np.array(Image.fromarray(mask).resize((new_w, new_h)))

    raw_image = np.array(Image.fromarray(raw_image).resize((new_w, new_h)))

    return raw_image, image, mask




def normalize(I):
    I = I.astype(np.float32)
    new_I = np.zeros(I.shape, dtype=np.uint8)
    if len(I.shape) >2: # rgb image
        num_channels = 3
        for k in range(num_channels):
            level = I[:,:,k]
            new_I[:,:,k] = (((level - level.min())/(level.max()-level.min()))*255).astype(np.uint8)
    else:
        level = I
        new_I = (((level - level.min())/(level.max()-level.min()))*255).astype(np.uint8)

    return new_I





def edge_detection_canny(image_gray,sig):
    edges = canny(image_gray, sigma=sig)#low_threshold=0.55, high_threshold=0.8)
    return edges
def edge_detection_otsu(image_gray,image_mask,index,m,n):
    #val = filters.threshold_otsu(image_gray[image_mask == 1])
    thresh = image_gray > 150

    dilated = morphology.dilation(thresh, morphology.square(3))

    edges = dilated - thresh

    index +=1
    plt.subplot(n,m,index)
    plt.imshow(thresh,cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    return edges,index
def get_image_variance(image,mask):
    pix_vec= image[mask == 1][:]
    return pix_vec.var()
def Hough_Transform(raw_image, input_image, image_mask, save_path, imname ):
    raw_image, input_image, image_mask =get_resized_image(raw_image, input_image, image_mask)
    edges_thresh = 3500
    sig_filt = 2
    edges_count = 10000
    ########## Boundary Extraction ############################################
    structure1 = np.ones((7,7), dtype=np.int32)
    im_erode = ndimage.binary_erosion(image_mask,structure1).astype(np.float32)
    im_dilate = ndimage.binary_dilation(image_mask,structure1).astype(np.float32)

    mask = image_mask.astype(np.float32)/255.0
    boundary1 = mask - im_erode
    boundary2 = im_dilate - mask

    boundary1[boundary1<0]=0
    boundary2[boundary2<0]=0
    boundary = boundary1 + boundary2
    while edges_count >= edges_thresh and sig_filt < 5.5 :

        image = filters.gaussian(input_image, sigma=sig_filt,channel_axis=2)


        lab = normalize(color.rgb2lab(image))

        image_gray = lab[:,:,1]

        all_edges = edge_detection_canny(image_gray,1)
        edges = all_edges.copy()
        tmp = 1.0 - boundary
        edges = edges.astype(np.float32) * tmp

        edges [tmp == 0.0]=0.0
        edges = edges.astype(bool)

        edges_count = edges.sum()

        if sig_filt > 2:
            print ('SIGMA INCREASED : ', sig_filt, edges_count)
        sig_filt += 0.5
        break
    structure1 = np.ones((3,3), dtype=np.int32)
    ed_dilate = ndimage.binary_dilation(edges,structure1).astype(np.float32)
    label_img = label(ed_dilate, connectivity=2)
    props = regionprops(label_img)
    conn_comp_before = len(props)
    count = 0
    edges_filtered = edges.copy()
    for i in range(len(props)):
        tit = [str(round(props[i].eccentricity,2)),str(round(props[i].solidity,2)),str(round(props[i].perimeter,2))]

        if props[i].eccentricity >= 0.98 or  props[i].perimeter > 600 or  props[i].perimeter < 25:  #area > 120:
            count +=1
            edges_filtered[props[i].coords[:,0],props[i].coords[:,1]] = 0

    conn_comp_after = conn_comp_before - count


    ########## Hough Transform ###########################################
    index =0
    n = 4
    m = 5 # num of cols
    hough_radii = [5,7,9,11,15]#np.arange(7, 12, 2)
    peaks_num = [1,15,15,10,2]
    hough_res = hough_circle(edges_filtered, hough_radii)
    #print 'Hough Res' , hough_res.shape
    centers = []
    accums = []
    radii = []
    indd=0
    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        num_peaks = peaks_num[indd]
        peaks = peak_local_max(h, num_peaks=num_peaks,min_distance=30)#,indices=False,threshold_rel=0.1)

        #print radius, peaks.shape
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])


        radii.extend([radius] * len(peaks))
        indd +=1

    I_top10 =  input_image.copy()
    for ind in np.argsort(accums)[::-1][:10]:
        center_x, center_y = centers[ind]
        radius = radii[ind]
        cx, cy = circle_perimeter(center_y, center_x, radius)

        try:
            I_top10[cy, cx] = (0, 0,255) #colors_p[ii]#(0, 0,255)
        except:
            pass


    I_top5 =  input_image.copy()
    for ind in np.argsort(accums)[::-1][:5]:
        center_x, center_y = centers[ind]
        radius = radii[ind]
        cx, cy = circle_perimeter(center_y, center_x, radius)

        try:
            I_top5[cy, cx] = (0, 0,255) #colors_p[ii]#(0, 0,255)
        except:
            pass
    I_top2 =  input_image.copy()
    for ind in np.argsort(accums)[::-1][:2]:
        center_x, center_y = centers[ind]
        radius = radii[ind]
        cx, cy = circle_perimeter(center_y, center_x, radius)

        try:
            I_top2[cy, cx] = (0, 0,255) #colors_p[ii]#(0, 0,255)
        except:
            pass

    I_all =  input_image.copy()
    for ind in np.argsort(accums)[::-1]:
        center_x, center_y = centers[ind]
        radius = radii[ind]
        cx, cy = circle_perimeter(center_y, center_x, radius)

        try:
            I_all[cy, cx] = (0, 0,255) #colors_p[ii]#(0, 0,255)
        except:
            pass


    subplot_flag = True
    imsave_flag = False
    if subplot_flag == True:
        plt.close('all')
        index +=1
        plt.subplot(n,m,index)
        plt.imshow(image,cmap=plt.cm.gray)
        plt.title('Sigma = ' + str(sig_filt-0.5),fontsize=2)
        plt.xticks([])
        plt.yticks([])


        index +=1
        plt.subplot(n,m,index)
        plt.imshow(image_gray,cmap=plt.cm.gray)
        plt.title('a* level',fontsize=2)
        plt.xticks([])
        plt.yticks([])


        index +=1
        plt.subplot(n,m,index)
        plt.imshow(all_edges,cmap=plt.cm.gray)
        plt.title('Canny Edges',fontsize=2)
        plt.xticks([])
        plt.yticks([])


        index +=1
        plt.subplot(n,m,index)
        plt.imshow(boundary,cmap=plt.cm.gray)
        plt.title('Body Boundary',fontsize=2)
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(edges,cmap=plt.cm.gray)
        plt.title('Count of Edges: '+str(edges_count),fontsize=2)
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(edges_filtered,cmap=plt.cm.gray)
        plt.title('Count of Edges: '+str(edges_filtered.sum()),fontsize=2)
        plt.xticks([])
        plt.yticks([])

        for radius, h in zip(hough_radii, hough_res):
            index +=1
            plt.subplot(n,m,index)
            plt.imshow(h,cmap=plt.cm.gray)
            plt.title('Radius = '+str(radius),fontsize=2)
            plt.xticks([])
            plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(I_top2)
        plt.title('Top 2 Circles',fontsize=2)
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(I_top5)
        plt.title('Top 5 Circles',fontsize=2)
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(I_top10)
        plt.title('Top 10 Circles',fontsize=2)
        plt.xticks([])
        plt.yticks([])

        index +=1
        plt.subplot(n,m,index)
        plt.imshow(I_all)
        plt.title('All Circles',fontsize=2)
        plt.xticks([])
        plt.yticks([])


        plt.savefig(save_path +imname +'_nipple_demo1.jpeg',bbox_inches='tight',dpi=300)
        plt.close('all')

        idx = 10
        if imsave_flag ==  True:
            plt.imsave(save_path +  str(idx)+'_1_' + imname + '.jpeg',image )
            plt.imsave(save_path +  str(idx)+'_2_' + imname + '.jpeg', image_gray,cmap=plt.cm.gray)
            plt.imsave(save_path +  str(idx)+'_3_' + imname + '.jpeg', all_edges,cmap=plt.cm.gray)
            plt.imsave(save_path +  str(idx)+'_4_' + imname + '.jpeg',boundary ,cmap=plt.cm.gray)
            plt.imsave(save_path +  str(idx)+'_5_' + imname + '.jpeg', edges,cmap=plt.cm.gray)
            plt.imsave(save_path +  str(idx)+'_6_' + imname + '.jpeg',edges_filtered ,cmap=plt.cm.gray)
            indexx =7
            for radius, h in zip(hough_radii, hough_res):
                plt.imsave(save_path +  str(idx)+'_'+str(indexx)+'_' + imname + '.jpeg',h,cmap=plt.cm.gray )
                indexx +=1
            plt.imsave(save_path +  str(idx)+'_13_' + imname + '.jpeg',I_top2 )
            plt.imsave(save_path +  str(idx)+'_14_' + imname + '.jpeg',I_top5 )
            plt.imsave(save_path +  str(idx)+'_15_' + imname + '.jpeg', I_top10)
            plt.imsave(save_path +  str(idx)+'_16_' + imname + '.jpeg',I_all )

        stats = [edges_count , edges_filtered.sum(), conn_comp_before, conn_comp_after,sig_filt]
        meas= False_Positive_Elimination(accums,centers,radii,input_image,save_path,imname,index)

        plt.imsave(save_path + imname + '_resized.jpeg', raw_image, cmap=plt.cm.gray)
        plt.close('all')


def False_Positive_Elimination(accums,centers,radii, image,save_path,imname,im_index):
    strt_plots_indx  = 17

    image_width = image.shape[1]
    image_height =  image.shape[0]
    min_chest_area_y = 0
    max_chest_area_y = 190
    y_diff_thresh = 15

    left_nipple_centers = []
    left_measures =[]
    left_radii = []

    right_nipple_centers = []
    right_measures = []
    right_radii = []

    #print len(accums),len(centers),len(radii)
    measures_list = []

    accums = np.array(accums)
    accums = accums / accums.max()

    template_nipple = color.rgb2gray(iio.imread('nipple.jpg')).astype(np.float64)
    template_nipple = (template_nipple - template_nipple.min()) / (template_nipple.max() - template_nipple.min())
    margin = 0
    image1 =  image.copy()
    idx = 0
    n =10
    m =10
    plt.close('all')
    idx +=1
    plt.subplot(n,m,idx)
    plt.imshow(template_nipple,cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    for i in range(len(centers)):
        center_y, center_x = centers[i]
        radius = radii[i]

        rgb_patch = image[center_y-radius-margin: center_y+radius+margin ,center_x-radius-margin:center_x+radius+margin,:]
        gray_patch = color.rgb2gray(rgb_patch).astype(np.float64)
        gray_patch = (gray_patch - gray_patch.min()) / (gray_patch.max() - gray_patch.min())

        tmp = template_nipple.copy()
        base_nipple = np.array(Image.fromarray(tmp).resize((gray_patch.shape[1],gray_patch.shape[0]))).astype(np.float64)


        #base_nipple = misc.imresize(tmp,(gray_patch.shape[0],gray_patch.shape[1])).astype(np.float64)/255.0

        #print(base_nipple.min(), gray_patch.min(), base_nipple.max(), gray_patch.max())
        S_meas = ssim(base_nipple,gray_patch,  data_range=gray_patch.max() - gray_patch.min())
        P_meas = accums[i]



        idx +=1
        plt.subplot(n,m,idx)
        plt.imshow(gray_patch,cmap=plt.cm.gray)
        plt.title('SSIM: %.2f, Peak: %.2f' % (S_meas,P_meas),fontsize=2)
        plt.xticks([])
        plt.yticks([])

        measures_list.append([imname, idx ,S_meas,P_meas])
        cx, cy = circle_perimeter(center_x, center_y, radius)

        #image1[cy,cx] = (0,255,0)
        if center_x < image_width//2 :
            #print center_y, center_x , 'Left'
            left_nipple_centers.append(centers[i])
            left_measures.append(S_meas + P_meas)
            left_radii.append(radii[i])


        elif center_x >= (image_width)/2 :
            #print center_y, center_x,'Right'
            right_nipple_centers.append(centers[i])
            right_measures.append(S_meas + P_meas)
            right_radii.append(radii[i])

    plt.subplots_adjust(hspace = .7)
    plt.savefig(save_path +imname +'_nipple_demo2.jpeg',bbox_inches='tight',dpi=1000)
    plt.close('all')
    strt_plots_indx +=1


    im =  image1.copy() # with the green circles
    nipples_mask = image1.copy()
    l_ind = len(left_measures)
    r_ind = len(right_measures)


    if( len(left_measures)>0):
        sorted_left_indices = np.argsort(left_measures)[::-1]
        l_ind = 0
        center_y, center_x = left_nipple_centers[sorted_left_indices[l_ind]]
        radius = left_radii[sorted_left_indices[l_ind]]
        cx, cy = circle_perimeter(center_x, center_y, radius)

        im[cy, cx] = (0, 0,255)

        cx, cy = circle((center_x, center_y), radius)
        nipples_mask[cy, cx]  = (0,0,0)

    if( len(right_measures)>0):
        sorted_right_indices = np.argsort(right_measures)[::-1]
        r_ind = 0
        center_y, center_x = right_nipple_centers[sorted_right_indices[r_ind]]
        radius = right_radii[sorted_right_indices[r_ind]]
        cx, cy = circle_perimeter(center_x, center_y, radius)
        im[cy, cx] = (0, 0,255)

        cx, cy = circle((center_x, center_y), radius)
        nipples_mask[cy, cx] = (0,0,0)

    plt.imsave(save_path + imname +'_no_nipples_stage1.jpeg', nipples_mask,cmap=plt.cm.gray)
    plt.close('all')

    im = image1.copy()
    nipples_mask = image1.copy()
    # step2: refinement method

    if l_ind < len(left_measures) and r_ind < len(right_measures) :
        l_center_y, l_center_x = left_nipple_centers[sorted_left_indices[l_ind]]
        r_center_y, r_center_x = right_nipple_centers[sorted_right_indices[r_ind]]
        y_diff = np.absolute(l_center_y - r_center_y)

    while y_diff > y_diff_thresh and l_ind < len(left_measures) and r_ind < len(right_measures):
        if np.absolute( max_chest_area_y - r_center_y) >  np.absolute( max_chest_area_y - l_center_y):
            #left_measures[sorted_left_indices[l_ind]] > right_measures[sorted_right_indices[r_ind]] :
            r_ind +=1
        else:
            l_ind +=1
            if l_ind < len(left_measures) and r_ind < len(right_measures) :
                l_center_y, l_center_x = left_nipple_centers[sorted_left_indices[l_ind]]
                r_center_y, r_center_x = right_nipple_centers[sorted_right_indices[r_ind]]
                y_diff = np.absolute(l_center_y - r_center_y)

    if( l_ind < len(left_measures)):
        center_y, center_x = left_nipple_centers[sorted_left_indices[l_ind]]
        radius = left_radii[sorted_left_indices[l_ind]]
        cx, cy = circle_perimeter(center_x, center_y, radius)
        im[cy, cx] = (0, 0,255)
        cx, cy = circle((center_x, center_y), radius)
        nipples_mask[cy, cx] = (0, 0, 0)

    if( r_ind < len(right_measures)):
        center_y, center_x = right_nipple_centers[sorted_right_indices[r_ind]]
        radius = right_radii[sorted_right_indices[r_ind]]
        cx, cy = circle_perimeter(center_x, center_y, radius)
        im[cy, cx] = (0, 0,255)
        cx, cy = circle((center_x, center_y), radius)
        nipples_mask[cy, cx] = (0, 0, 0)

    plt.imsave(save_path + imname + '_no_nipples_stage2.jpeg', nipples_mask, cmap=plt.cm.gray)
    plt.close('all')



    return measures_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path', default="../sample_images/img_trunk.jpg", type=str)
    parser.add_argument('--save_path', default="../sample_results/nipple/", type=str)
    args = parser.parse_args()

    image, skin_mask, foreground = segment_image(args.im_path, None)


    imname = args.im_path.split('/')[-1][:-4]
    Hough_Transform(image, foreground, skin_mask, args.save_path, imname)
    ## Sample Example
    ## python nipple_detection.py --im_path ../sample_images/img_trunk2.JPG --save_path ../sample_results/nipple/
