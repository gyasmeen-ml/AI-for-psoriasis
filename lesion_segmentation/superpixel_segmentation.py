'''
Author: Dr Yasmeen George
Email: yasmeen.mourice@gmail.com
Date: 24 May 2023
'''
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn import linear_model
from sklearn import tree
import segmentation_slic as helper
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.io import imread
from PIL import Image
from skimage.segmentation import mark_boundaries
import csv
from py_img_seg_eval.eval_segm import *
from scipy.ndimage import binary_erosion
import time
import imageio.v3 as iio
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

import warnings
warnings.filterwarnings("ignore")


def read_csv_file(f_name):
    with open(f_name,'r') as dest_f:
        reader = csv.reader(dest_f)#, delimiter=',',quotechar = '"')
        next(reader, None)  # skip the headers
        data = [data for data in reader]
        #print data

    data_array = np.asarray(data, dtype = np.float32)
    #print data_array.dtype, data_array.shape
    return data_array



def read_superpixels_features():
    train_data = read_csv_file('train_superpixel_features.csv')
    test_data = read_csv_file('test_superpixel_features.csv')

    SuperPixel_X_train = train_data[:,1:60]
    SuperPixel_X_test = test_data[:,1:60]

    SuperPixel_y_train = train_data[:,61]
    SuperPixel_y_test = test_data[:,61]
    return SuperPixel_X_train, SuperPixel_X_test, SuperPixel_y_train, SuperPixel_y_test

def lin_regression(SuperPixel_X_train, SuperPixel_X_test, SuperPixel_y_train, SuperPixel_y_test):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(SuperPixel_X_train, SuperPixel_y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"   % np.mean((regr.predict(SuperPixel_X_test) - SuperPixel_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(SuperPixel_X_test, SuperPixel_y_test))

    # Plot outputs
    plt.scatter(SuperPixel_X_test[:,55], SuperPixel_y_test,  color='black')
    plt.plot(SuperPixel_X_test[:,55], regr.predict(SuperPixel_X_test), color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.savefig('psoriasis_regression.png')


def tree_regression(SuperPixel_X_train, SuperPixel_X_test, SuperPixel_y_train, SuperPixel_y_test):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(SuperPixel_X_train, SuperPixel_y_train)
    clf.predict(SuperPixel_X_test)
    print("Mean squared error: %.2f"   % np.mean((clf.predict(SuperPixel_X_test) - SuperPixel_y_test) ** 2))
    print('Variance score: %.2f' % clf.score(SuperPixel_X_test, SuperPixel_y_test))
    return clf

def tree_classifier(SuperPixel_X_train, SuperPixel_X_test, SuperPixel_y_train, SuperPixel_y_test):

    SuperPixel_y_train = SuperPixel_y_train >= 0.5
    SuperPixel_y_test = SuperPixel_y_test >=0.5

    print (SuperPixel_y_train.dtype , SuperPixel_y_train.min() , SuperPixel_y_train.max())
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(SuperPixel_X_train, SuperPixel_y_train)
    clf.predict(SuperPixel_X_test)
    print("Mean squared error: %.2f"   % np.mean((clf.predict(SuperPixel_X_test) - SuperPixel_y_test) ** 2))
    print('Variance score: %.2f' % clf.score(SuperPixel_X_test, SuperPixel_y_test))
    return clf

def cluster_kmeans(SuperPixel_X_train):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(SuperPixel_X_train)
    #kmeans.labels_
    #kmeans.cluster_centers_

    return kmeans
def remove_background_segments(features_mat,labels, segments):
    indices = []
    num_segments = segments.max()+1
    new_segments =  np.ones(segments.shape, np.int32) * (num_segments+1)
    for i in range (num_segments):
        #print(i, features_mat[i, 0] ,features_mat[i, 1] ,features_mat[i, 2])
        if features_mat[i,0] > 0.1 and features_mat[i,1] > 0.1 and features_mat[i,2] > 0.1:
            indices.append(i)
            tmp = np.where(segments == i)
            y_ind = tmp[0]
            x_ind = tmp[1]
            new_segments[y_ind,x_ind]= i
    indices = np.asarray(indices,np.int32)
    return features_mat[indices,:],labels[indices] ,new_segments, indices

def segment_image(img ,gt,numSegments):

    image = img_as_float(img)

    start = time.time()
    all_segments = slic(image, n_segments = numSegments, sigma = 5)
    end = time.time()
    slic_time = end - start #####################################################################

    features_mat,labels = helper.get_segments_features(all_segments , image, gt,smooth_flag=False,sig_filt=3)#####################



    #print features_mat.shape
    start = time.time()
    features_mat,labels ,segments, indices = remove_background_segments(features_mat,labels, all_segments)
    end = time.time()
    suppre_time = end - start #####################################################################



    rgb_ind = [0,1,2,12,13,14,24,25,26,36,37,38,48,49,50]
    ycbcr_ind = [3,4,5,15,16,17,27,28,29,39,40,41,51,52,53]
    hsv_ind =[6,7,8,18,19,20,30,31,32,42,43,44,54,55,56]
    lab_ind = [9,10,11,21,22,23,33,34,35,45,46,47,57,58,59]
    ab_ind = [10,11,22,23,34,35,46,47,58,59]
    a_mean_ind = 4
    X_test = features_mat[:,ab_ind]


    #a_mean_ind = 34
    #X_test = features_mat[:,:]

    y_test = labels
    #print y_test.shape , y_test.dtype	,y_test.min(),y_test.max()
    y_test = np.asarray(y_test.T,np.int32)


    mi_vector = mutual_info_classif(X_test,y_test,discrete_features=True)
    #mi_vector = np.asarray(mi_vector , np.float32)

    f_vector = f_classif(X_test,y_test)
    #f_vector = np.asarray(f_vector,np.float32)

    start = time.time()
    clf = cluster_kmeans(X_test)

    y_predict = clf.predict(X_test)
    end = time.time()
    kmeans_time = end - start #####################################################################
 
    centers= clf.cluster_centers_

    #print("Mean squared error: %.2f"   % np.mean((y_predict - y_test) ** 2))

    num_segments = segments.max()+1
    segment_res = np.zeros(segments.shape,np.float32)
    binary_seg = np.zeros(segments.shape,np.float32)
    score_img = np.zeros(segments.shape,np.float32)
    ind = 0
    for i in indices:
        tmp = np.where(segments == i)
        y_ind = tmp[0]
        x_ind = tmp[1]

        segment_res[y_ind,x_ind]=y_predict[ind]+1
        if centers[0][a_mean_ind] > centers[1][a_mean_ind]:  # define lesion class based on a values of computed centers
            binary_seg[y_ind,x_ind]=1-y_predict[ind]
        else:
            binary_seg[y_ind,x_ind]=y_predict[ind]

        score_img[y_ind,x_ind] = labels[ind]
        ind = ind+1

    super_pixels_image= mark_boundaries(image, all_segments,(0,0,1))
    super_pixels_filtered_image= mark_boundaries(image, segments,(0,0,1))
    gray_segmented_image = np.asarray((segment_res*255)/segment_res.max() , np.uint8)
    score_img = np.asarray(score_img*255 , np.uint8)

    pa = pixel_accuracy(binary_seg, gt)
    ma = mean_accuracy(binary_seg, gt)
    m_IU = mean_IU(binary_seg, gt)
    fw_IU = frequency_weighted_IU(binary_seg, gt)
    #stats_vector = (mi_vector, f_vector[1],image.shape[0], image.shape[1],pa,ma,m_IU,fw_IU)

    stats_vector = []
    for x in mi_vector:
        stats_vector.append(x)
    for x in f_vector[1]:
        #print x
        stats_vector.append(x)
    for x in [segment_res.shape[0], segment_res.shape[1],numSegments,pa,ma,m_IU,fw_IU]:
        stats_vector.append(x)

    stats_vector = np.asarray(stats_vector,np.float32)


    binary_seg = img_as_float(binary_seg)
    gt1 = binary_seg - binary_erosion(binary_seg,structure=np.ones((5,5)))
    tmp = np.where(gt1 == 1)
    y_ind = tmp[0]
    x_ind = tmp[1]
    rgb_segmentedimage = image.copy()
    rgb_segmentedimage[y_ind,x_ind,:] = (0,0,1)
    vis_results= [super_pixels_image ,super_pixels_filtered_image, score_img,binary_seg,gray_segmented_image,rgb_segmentedimage]

    return vis_results, stats_vector, np.asarray(binary_seg,np.int32),[slic_time,suppre_time,kmeans_time]

  
  
def run_batch(save_stats=False,save_figs = False,file_names=None, Save_Path='' ):


    import csv
    if save_stats == True:
        myfile1 = open('segmentation_sp_accuracy.csv', 'wb')
        wr1 = csv.writer(myfile1, quoting=csv.QUOTE_ALL)

    if file_names == None:
        return

    images, labels = helper.create_full_paths(file_names, '/home/ygeorge/PsoriasisDatasets/LesionSegmentation-sp/All_Images/', '/home/ygeorge/PsoriasisDatasets/LesionSegmentation-sp/All_Images/BW_')

    accuracy =  np.zeros((len(file_names),4),np.float32)
    exec_time =  np.zeros((len(file_names),23),np.float32)
    for index in range(len(file_names)):
        fus_time = 0
        #print (index+1 , 'out of ', len(file_names))
        img = iio.imread(images[index])
        gt = iio.imread(labels[index])

        count = 1
        fused_image = np.zeros(gt.shape,np.float32)
        header = []
        accuracy_stats = []

        tt = 0
        for S in [40,30,25,20,15]:
            S2 = S * S
            N = img.shape[0] * img.shape[1]
            num_seg = int(N / S2)
            vis_results,accuracy_vec,binary_seg,time_exec = segment_image(img ,gt,num_seg)
            pa,j_index , dc = Similarity_Metrics(binary_seg, gt)
            print ( S ,  pa*100 , j_index , dc)
            exec_time[index,tt:tt+4]= time_exec
            tt= tt+4

            if save_figs == True:
                for Img in vis_results:
                    if len(Img.shape)<3:
                        plt.imsave(Save_Path +  file_names[index] + '_'+str(count)+'.jpg',Img,cmap = plt.cm.gray)
                    else:
                        plt.imsave(Save_Path +  file_names[index] + '_'+str(count)+'.jpg',Img)
            count = count +1
            start = time.time()
            fused_image = fused_image + binary_seg
            end = time.time()
            fus_time = fus_time + (end - start) ####################################################

        if save_stats == True:
            if index == 0 : # write HEADER
                meas = ['MI', 'F']
                channels = ['A','B']
                #channels = ['R', 'G', 'B' , 'Y', 'Cb' , 'Cr', 'H', 'S','V','L','A' , 'B']
                funcs = ['min', 'max', 'mean', 'var', 'entropy']

                for m in meas:
                    for f in funcs:
                        for ch in channels:
                            val = m + '_'+ ch + '_'+ f
                            header.append(val)
                for v in ['Height', 'Width', 'K','pixel_accuracy','mean_accuracy','mean_IU','freq_weighted_IU']:
                    header.append(v)
            accuracy_stats = np.append(accuracy_stats, accuracy_vec)

        tit = ''
        plt.imsave(Save_Path + tit +file_names[index] + '_'+str(count)+'.jpg',img)
        count = count +1
        plt.imsave(Save_Path +  tit + file_names[index] + '_'+str(count)+'.jpg',gt,cmap = plt.cm.gray)
        count = count+1
        plt.imsave(Save_Path + tit + file_names[index] + '_'+str(count)+'.jpg',np.asarray((fused_image*255)/fused_image.max()),cmap = plt.cm.gray)
        count = count+1


        maj_fused_image = fused_image >2
        end = time.time()
        fus_time = fus_time + (end - start)
        exec_time[index,20]=fus_time
        exec_time[index,21]=gt.shape[0]
        exec_time[index,22]=gt.shape[1]

        pa,j_index , dc = Similarity_Metrics(maj_fused_image, gt)
        #print ('fused: ', pa*100, j_index,dc)
        accuracy[index,0] = index
        accuracy[index,1] = pa
        accuracy[index,2] = j_index
        accuracy[index,3] = dc

        maj_fused_image = img_as_float(maj_fused_image)
        gt1 = maj_fused_image - binary_erosion(maj_fused_image,structure=np.ones((5,5)))
        tmp = np.where(gt1 == 1)
        y_ind = tmp[0]
        x_ind = tmp[1]
        final_segmented_image = img.copy()
        final_segmented_image[y_ind,x_ind,:] = (0,0,1)

        final_segmented_image_blue = img.copy()
        final_segmented_image_blue[y_ind,x_ind,:] = (0,0,255)

        #final_segmented_image= mark_boundaries(img, maj_fused_image,(0,0,1))
        plt.imsave(Save_Path +  tit + file_names[index] + '_'+str(count)+'.jpg',maj_fused_image,cmap = plt.cm.gray)
        count = count +1
        plt.imsave(Save_Path +  tit +file_names[index] + '_'+str(count)+'.jpg',final_segmented_image)
        count = count +1
        plt.imsave(Save_Path +  tit + file_names[index] + '_'+str(count)+'.jpg',final_segmented_image_blue)

        final_segmented_image_blue[y_ind,x_ind,:] = (0,0,153)
        count = count +1
        plt.imsave(Save_Path +  tit + file_names[index] + '_'+str(count)+'.jpg',final_segmented_image_blue)

        final_segmented_image_blue[y_ind,x_ind,:] = (102, 0, 204)
        count = count +1
        plt.imsave(Save_Path +  tit + file_names[index] + '_'+str(count)+'.jpg',final_segmented_image_blue)


        if save_stats == True:
            if index == 0:
                wr1.writerow(header)

            wr1.writerow(accuracy_stats)

            np.savetxt("sp_accuracy_measures.csv", accuracy, delimiter=",")
            np.savetxt("sp_exec_time.csv",  exec_time, delimiter=",")

    if save_stats == True:
        myfile1.close()

def run(im_path, mask_path,  save_path, imname ):


    img = imread(im_path)
    gt = imread(mask_path)

    resize = True
    if len(gt.shape) > 2:
        gt = gt[:,:,0]

    if gt.shape[0] != img.shape[0]:
        gt = np.array(Image.fromarray(gt).resize((img.shape[1], img.shape[0]))).astype(np.float64)
    if resize:
        d_h = 640
        ratio = d_h / float(img.shape[0])
        d_w = int(ratio*img.shape[1])
        gt = np.array(Image.fromarray(gt).resize((d_w, d_h)))
        img = np.array(Image.fromarray(img).resize((d_w, d_h)))

    gt[gt > 0] = 255

    fused_image = np.zeros(gt.shape,np.float32)
    tt = 0
    save_figs=False
    count = 1
    for S in [40,30,25,20,15]:
        S2 = S * S
        N = img.shape[0] * img.shape[1]
        num_seg = int(N / S2)
        vis_results,accuracy_vec,binary_seg,time_exec = segment_image(img ,gt,num_seg)

        pa,j_index , dc = Similarity_Metrics(binary_seg, gt)
        #print ( S ,  pa*100 , j_index , dc)

        tt= tt+4

        if save_figs == True:
            for Img in vis_results:
                if len(Img.shape)<3:
                    plt.imsave(save_path + imname + '_'+str(count)+'.jpg',Img,cmap = plt.cm.gray)
                else:
                    plt.imsave(save_path + imname +'_'+str(count)+'.jpg',Img)
        count = count +1

        fused_image = fused_image + binary_seg

    if save_figs == True:
        plt.imsave(save_path + imname+ '_'+str(count)+'.jpg',img)
        count = count +1
        plt.imsave(save_path + imname + '_'+str(count)+'.jpg',gt,cmap = plt.cm.gray)
        count = count+1
        plt.imsave(save_path + imname + '_'+str(count)+'.jpg',np.asarray((fused_image*255)/fused_image.max()),cmap = plt.cm.gray)


    maj_fused_image = fused_image >2
    pa,j_index , dc = Similarity_Metrics(maj_fused_image, gt)
    #print ('fused: ', pa*100, j_index,dc)

    maj_fused_image = img_as_float(maj_fused_image)
    gt1 = maj_fused_image - binary_erosion(maj_fused_image,structure=np.ones((5,5)))
    tmp = np.where(gt1 == 1)
    y_ind = tmp[0]
    x_ind = tmp[1]
    final_segmented_image = img.copy()
    final_segmented_image[y_ind,x_ind,:] = (0,0,1)

    final_segmented_image_blue = img.copy()
    final_segmented_image_blue[y_ind,x_ind,:] = (0,0,255)

    plt.imsave(save_path + imname + '_lesion_mask.jpg',maj_fused_image,cmap = plt.cm.gray)
    final_segmented_image_blue[y_ind,x_ind,:] = (0,0,153)
    plt.imsave(save_path + imname + '_lesion_segmentation.jpg',final_segmented_image_blue)



def jaccard_coefficient(y_true,y_pred):

    return jaccard_score(y_true.flatten(), y_pred.flatten())
    #return jaccard_similarity_score(y_true, y_pred, normalize=False)
def Similarity_Metrics(img1,img2):

    img1 = np.asarray(img1).astype(bool)
    img2 = np.asarray(img2).astype(bool)

    pa = pixel_accuracy(img1, img2)
    ################JACCARDS ##############################
    #Find the intersection of the two images
    inter_image = np.logical_and(img1, img2)
    #Find the union of the two images
    union_image = np.logical_or(img1,img2)

    inter_sum =  inter_image.sum().astype(np.float32)
    union_sum = union_image.sum().astype(np.float32)
    j_ind = inter_sum/union_sum


    img1_sum = img1.sum().astype(np.float32)
    img2_sum = img2.sum().astype(np.float32)

    dice_coef = (2 * inter_sum) / (img1_sum + img2_sum)
    return pa,j_ind , dice_coef
def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0


    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2 * intersection.sum() / im_sum
'''
F=['IM58_UR_2131413_KENA_20130730_13H','IM59_UR_2131413_KENA_20130730_13H','IM60_UR_2131413_KENA_20130926_04H','IM73_UR_2131413_KENA_20130926_06H','IM81_UR_2131413_KENA_20130926_08H','IM82_UR_2131413_KENA_20130926_10H','IM87_UR_2131413_KENA_20130926_10H','IM91_UR_2131413_KENA_20130926_10H','IM100_UR_2131413_KENA_20130926_12H','IM103_UR_2131413_KENA_20130926_12H','IM2_UR_849783_SCHK_20160301_03','IM13_UR_1119661_HAIS_20130813_07H','IM20_UR_1119661_HAIS_20130813_07H','IM34_UR_1119661_HAIS_20130813_13H','IM50_UR_1119661_HAIS_20131107_07H','IM55_UR_1119661_HAIS_20131107_07H','IM68_UR_1119661_HAIS_20131107_13H','IM23_UR_682655_KOUA_20121025_07H','IM35_UR_682655_KOUA_20121025_07H','IM45_UR_682655_KOUA_20121025_09H','IM52_UR_682655_KOUA_20121025_09H','IM94_UR_682655_KOUA_20121025_13H','IM17_UR_1384545_KUPW_20100422_04H_Amgen','IM73_UR_1384545_KUPW_20100429_09H','IM104_UR_1384545_KUPW_20100506_10H','IM7_UR_1335182_MJ_20081007_10','IM9_UR_1335182_MJ_20081007_10','IM23_UR_1335182_MJ_20081007_11','IM30_UR_1335182_MJ_20081007_11','IM15_UR_1119661_HAIS_20130813_07H','IM30_UR_2001339_GONL_20141216_08','IM27_UR_2001339_GONL_20141216_08']

F = ['IM1_UR_2040723_FANQ_20150106_01'] #color spaces

F2= ['IM3_UR_2272548_DIMV_20160209_06','IM11_UR_2272548_DIMV_20160209_02','IM15_UR_1199302_NGUQ_20150922_06','IM12_UR_559397_YOUT_20121101_09H','IM15_UR_2001339_GONL_20141216_05','IM14_UR_1199302_NGUQ_20150922_05','IM8_UR_559397_YOUT_20121101_07H','IM5_UR_2206068_COLJ_20141007_02']

F3= ['IM28_UR_2001339_GONL_20141216_08','IM1_UR_559397_YOUT_20121101_07H','IM1_UR_682655_KOUA_20121025_05H']
F3=['IM2_UR_2040723_FANQ_20150106_01']
run_batch(save_stats=False,save_figs = False,file_names=F3, Save_Path = '/data/projects/punim0066/Segmentation/Superpixels/tmp_superpixel_results/' )


#main(save_stats=True,save_figs = True,file_names=None, Save_Path = '/data/projects/punim0066/Segmentation/Superpixels/tmp_superpixel_results_FINAL/' )
'''


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path', default="../sample_images/img_lesion.jpg", type=str)
    parser.add_argument('--mask_path', default="../sample_images/img_lesion_mask.jpg", type=str)
    parser.add_argument('--save_path', default="../sample_results/lesion/", type=str)
    args = parser.parse_args()

    imname = args.im_path.split('/')[-1][:-4]
    run(args.im_path,args.mask_path, args.save_path, imname)
    ## Sample Example
    ## python superpixel_segmentation.py --im_path ../sample_images/img_trunk2.jpg --mask_path "../sample_results/nipple/img_trunk2_no_nipples_stage2.jpeg" --save_path ../sample_results/lesion/