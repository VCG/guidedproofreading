from backport_collections import Counter
import imread
import numpy as np
import mahotas as mh
import random
from scipy.optimize import curve_fit
from scipy.spatial import distance
import skimage.measure
import tifffile as tif
import time


from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt    


from patch import Patch
from util import Util

class Legacy(object):

  @staticmethod
  def read_dojo_data():
    input_image = np.zeros((10,1024,1024))
    input_rhoana = np.zeros((10,1024,1024))
    input_gold = np.zeros((10,1024,1024))
    input_prob = np.zeros((10,1024,1024))
    path_prefix = '/Users/d/Projects/'
    path_prefix = '/home/d/dojo_xp/data/' # for beast only
    # input_rhoana_ = imread.imread_multi(path_prefix+'dojo_data_vis2014/labels_after_automatic_segmentation_multi.tif')
    input_rhoana = tif.imread(path_prefix+'dojo_data_vis2014/labels_after_automatic_segmentation_multi.tif')
    # for i,r in enumerate(input_rhoana_):
    #   input_rhoana[i] = r

    input_gold_ = imread.imread_multi(path_prefix+'dojo_data_vis2014/groundtruth_multi.tif')
    for i,g in enumerate(input_gold_):
      input_gold[i] = g 
    for i in range(10):
        # input_prob[i] = mh.imread(path_prefix+'dojo_data_vis2014/prob/'+str(i)+'_syn.tif')
        input_prob[i] = tif.imread(path_prefix+'dojo_data_vis2014/prob_unet_fixed/'+str(50+i).zfill(4)+'.tif')
        # return prob
        # input_prob[i] = np.rot90(np.pad(prob, 92, 'constant'),2) # we are rotating the image
        input_image[i] = mh.imread(path_prefix+'dojo_data_vis2014/images/'+str(i)+'.tif')
        
    bbox = mh.bbox(input_image[0])

    bbox_larger = [bbox[0]-37, bbox[1]+37, bbox[2]-37, bbox[3]+37]

    prob_new = np.zeros(input_image.shape, dtype=np.uint8)
    
    input_image = input_image[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_rhoana = input_rhoana[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_gold = input_gold[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    # input_prob = input_prob[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    
    #inverting right here
    prob_new[:,bbox[0]:bbox[1], bbox[2]:bbox[3]] = 255-input_prob[:,bbox[0]:bbox[1], bbox[2]:bbox[3]]
    prob_new = prob_new[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]



    for i in range(0,10):
      zeros_gold = Util.threshold(input_gold[i], 0)
      input_gold[i] = Util.relabel(input_gold[i])
      # restore zeros
      input_gold[i][zeros_gold==1] = 0
      input_rhoana[i] = Util.relabel(input_rhoana[i])

    return input_image.astype(np.uint8), prob_new.astype(np.uint8), input_gold.astype(np.uint32), input_rhoana.astype(np.uint32), bbox_larger

  @staticmethod
  def read_dojo_test_data():
    input_image = np.zeros((10,1024,1024))
    input_rhoana = np.zeros((10,1024,1024))
    input_gold = np.zeros((10,1024,1024))
    input_prob = np.zeros((10,1024,1024))
    path_prefix = '/Users/d/Projects/'
    path_prefix = '/home/d/dojo_xp/data/' # for beast only
    # input_rhoana_ = imread.imread_multi(path_prefix+'dojo_data_vis2014/labels_after_automatic_segmentation_multi.tif')
    input_rhoana = tif.imread(path_prefix+'dojo_data_vis2014_test/cut-train-segs.tif')
    # for i,r in enumerate(input_rhoana_):
    #   input_rhoana[i] = r
    print 'test data'
    input_gold_ = imread.imread_multi(path_prefix+'dojo_data_vis2014_test/cut-train-labels.tif')
    for i,g in enumerate(input_gold_):
      input_gold[i] = g 
    for i in range(10):
        # input_prob[i] = mh.imread(path_prefix+'dojo_data_vis2014/prob/'+str(i)+'_syn.tif')
        #
        # WE USE THE OTHER UNET PROBS, WE DO NOT NEED THEM HERE
        #
        input_prob[i] = tif.imread(path_prefix+'dojo_data_vis2014/prob_unet_fixed/'+str(50+i).zfill(4)+'.tif')
        # return prob
        # input_prob[i] = np.rot90(np.pad(prob, 92, 'constant'),2) # we are rotating the image
        input_image[i] = mh.imread(path_prefix+'dojo_data_vis2014_test/images/'+str(i)+'.tif')
        
    bbox = mh.bbox(input_image[0])

    bbox_larger = [bbox[0], bbox[1], bbox[2], bbox[3]]

    prob_new = np.zeros(input_image.shape, dtype=np.uint8)
    
    input_image = input_image[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_rhoana = input_rhoana[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_gold = input_gold[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    # input_prob = input_prob[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    
    #inverting right here
    prob_new[:,bbox[0]:bbox[1], bbox[2]:bbox[3]] = 255-input_prob[:,bbox[0]:bbox[1], bbox[2]:bbox[3]]
    prob_new = prob_new[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]



    for i in range(0,10):
      zeros_gold = Util.threshold(input_gold[i], 0)
      input_gold[i] = Util.relabel(input_gold[i])
      # restore zeros
      input_gold[i][zeros_gold==1] = 0
      input_rhoana[i] = Util.relabel(input_rhoana[i])

    # right now we have a 400x400 but we need 474x474
    input_image = np.pad(input_image, 37, mode='constant', constant_values=0)[37:-37]
    prob_new = np.pad(prob_new, 37, mode='constant', constant_values=0)[37:-37]
    input_gold = np.pad(input_gold, 37, mode='constant', constant_values=0)[37:-37]
    input_rhoana = np.pad(input_rhoana, 37, mode='constant', constant_values=0)[37:-37]








    return input_image.astype(np.uint8), prob_new.astype(np.uint8), input_gold.astype(np.uint32), input_rhoana.astype(np.uint32), bbox_larger


  @staticmethod
  def invert(array, smooth=False, sigma=2.5):
    
    grad = mh.gaussian_filter(array, sigma)

    return (255-grad)

  @staticmethod
  def gradient(array, sigma=5.5):
    '''
    '''

    grad = mh.gaussian_filter(array, sigma)

    grad_x = np.gradient(grad)[0]
    grad_y = np.gradient(grad)[1]
    grad = np.sqrt(np.add(grad_x*grad_x, grad_y*grad_y))

    grad -= grad.min()
    grad /= (grad.max() - grad.min())
    grad *= 255

    return grad


  @staticmethod
  def random_watershed(array, speed_image, border_seeds=False, erode=False):
    '''
    '''
    copy_array = np.array(array, dtype=np.bool)

    if erode:
      
      for i in range(10):
        copy_array = mh.erode(copy_array)


    seed_array = np.array(copy_array)
    if border_seeds:
      seed_array = mh.labeled.border(copy_array, 1, 0, Bc=mh.disk(7))

    coords = zip(*np.where(seed_array==1))

    if len(coords) == 0:
      # print 'err'
      return np.zeros(array.shape)

    seed1_ = None
    seed2_ = None
    max_distance = -np.inf

    for i in range(10):
      seed1 = random.choice(coords)
      seed2 = random.choice(coords)
      d = distance.euclidean(seed1, seed2)
      if max_distance < d:
        max_distance = d
        seed1_ = seed1
        seed2_ = seed2

    seeds = np.zeros(array.shape, dtype=np.uint8)
    seeds[seed1_[0], seed1_[1]] = 1
    seeds[seed2_[0], seed2_[1]] = 2



    for i in range(8):
      seeds = mh.dilate(seeds)

    # Util.view(seeds,large=True)      
    # print speed_image.shape, seeds.shape
    ws = mh.cwatershed(speed_image, seeds)
    ws[array == 0] = 0

    #return seeds, ws    
    return ws


  @staticmethod
  def fix_single_merge(cnn, cropped_image, cropped_prob, cropped_binary, N=10, invert=True, dilate=True, 
                       border_seeds=True, erode=False, debug=False, before_merge_error=None,
                       real_border=np.zeros((1,1)), oversampling=False, crop=True):
    '''
    invert: True/False for invert or gradient image
    '''

    bbox = mh.bbox(cropped_binary)

    orig_cropped_image = np.array(cropped_image)
    orig_cropped_prob  = np.array(cropped_prob)
    orig_cropped_binary = np.array(cropped_binary)



    speed_image = None
    if invert:
      speed_image = Legacy.invert(cropped_image, smooth=True, sigma=2.5)
    else:
      speed_image = Legacy.gradient(cropped_image)

    # Util.view(speed_image, large=False, color=False)


    dilated_binary = np.array(cropped_binary, dtype=np.bool)
    if dilate:
      for i in range(20):
          dilated_binary = mh.dilate(dilated_binary)      

    # Util.view(dilated_binary, large=False, color=False)

    borders = np.zeros(cropped_binary.shape)

    best_border_prediction = np.inf
    best_border_image = np.zeros(cropped_binary.shape)

    original_border = mh.labeled.border(cropped_binary, 1, 0, Bc=mh.disk(3))

    results_no_border = []
    predictions = []
    borders = []
    results = []

    for n in range(N):
        ws = Legacy.random_watershed(dilated_binary, speed_image, border_seeds=border_seeds, erode=erode)
        
        if ws.max() == 0:
          continue

        ws_label1 = ws.max()
        ws_label2 = ws.max()-1
        border = mh.labeled.border(ws, ws_label1, ws_label2)

        # Util.view(ws, large=True)


        # Util.view(border, large=True)

        # print i, len(border[border==True])

        #
        # remove parts of the border which overlap with the original border
        #

        

        ws[cropped_binary == 0] = 0

        # Util.view(ws, large=False, color=False)        

        ws_label1_array = Util.threshold(ws, ws_label1)
        ws_label2_array = Util.threshold(ws, ws_label2)

        eroded_ws1 = np.array(ws_label1_array, dtype=np.bool)
        eroded_ws2 = np.array(ws_label2_array, dtype=np.bool)
        if erode:

          for i in range(5):
            eroded_ws1 = mh.erode(eroded_ws1)

          # Util.view(eroded_ws, large=True, color=False)

          dilated_ws1 = np.array(eroded_ws1)
          for i in range(5):
            dilated_ws1 = mh.dilate(dilated_ws1)


          for i in range(5):
            eroded_ws2 = mh.erode(eroded_ws2)

          # Util.view(eroded_ws, large=True, color=False)

          dilated_ws2 = np.array(eroded_ws2)
          for i in range(5):
            dilated_ws2 = mh.dilate(dilated_ws2)




          new_ws = np.zeros(ws.shape, dtype=np.uint8)
          new_ws[dilated_ws1 == 1] = ws_label1
          new_ws[dilated_ws2 == 1] = ws_label2


          ws = new_ws

          # Util.view(new_ws, large=True, color=True)

        # ws[original_border == 1] = 0
        
        prediction = Patch.grab_group_test_and_unify(cnn, cropped_image, cropped_prob, ws, ws_label1, ws_label2, oversampling=oversampling)
        
        if prediction == -1 or prediction >= .5:
          # invalid
          continue


        # here we have for one border
        # the border
        # the prediction
        # borders.append(border)
        # predictions.append(prediction)
        results.append((prediction, border))



    return results




  @staticmethod
  def get_top5_merge_errors(cnn, input_image, input_prob, input_rhoana, verbose=False):

    #
    # this creates the top bins for the best five merge splits
    #
    t0 = time.time()
    fixed_volume = np.array(input_rhoana)

    merge_errors = []



    for i in range(len(input_image)):
        if verbose:
          print 'working on slice', i
        
        DOJO_SLICE = i
        
        hist = Util.get_histogram(input_rhoana[DOJO_SLICE].astype(np.uint64))
        labels = range(len(hist))

        fixed_slice = np.array(input_rhoana[DOJO_SLICE], dtype=np.uint64)

        for l in labels:

            if l == 0 or hist[l]<3000:
                continue

            # single binary mask for label l
            before_merge_error = np.zeros(input_rhoana[DOJO_SLICE].shape)
            before_merge_error[fixed_slice == l] = 1

            results = Legacy.fix_single_merge(cnn,
                                              input_image[DOJO_SLICE],
                                              input_prob[DOJO_SLICE],
                                              before_merge_error, N=50, 
                                              erode=True, 
                                              invert=True,
                                              dilate=True,
                                              border_seeds=True,
                                              oversampling=False)

            if len(results) > 0:
                
                #
                # SORT THE PREDICTIONS (prediction, border)-tupels
                # LOOK AT TOP 5
                sorted_pred = sorted(results, key=lambda x: x[0])

                top5 = sorted_pred[:5]
                
                lowest_prediction = sorted_pred[0][0]
                

                # store the merge error
                # we need to store: z, l, results_no_border, borders, predictions
                merge_errors.append((i, l, lowest_prediction, (top5)))
                
    if verbose:
      print 'merge error correction done after',time.time()-t0, 'seconds'

    return merge_errors



  @staticmethod
  def get_merge_error_image(input_image, input_rhoana, label, border):

    binary = Util.threshold(input_rhoana, label)
    binary_dilated = mh.dilate(binary.astype(np.bool))
    for dilate in range(30):
      binary_dilated = mh.dilate(binary_dilated)


    binary_bbox = mh.bbox(binary_dilated)
    binary_border = mh.labeled.borders(binary)

    b = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    b[:,:,0] = input_image[:]
    b[:,:,1] = input_image[:]
    b[:,:,2] = input_image[:]
    b[:,:,3] = 255

    c = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    c[:,:,0] = input_image[:]
    c[:,:,1] = input_image[:]
    c[:,:,2] = input_image[:]
    c[:,:,3] = 255        
    c[binary_border == 1] = (0,255,0,255)

    e = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    e[:,:,0] = input_image[:]
    e[:,:,1] = input_image[:]
    e[:,:,2] = input_image[:]
    e[:,:,3] = 255        

    f = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    f[:,:,0] = input_image[:]
    f[:,:,1] = input_image[:]
    f[:,:,2] = input_image[:]
    f[:,:,3] = 255  
    f[binary == 1] = (0,255,0,255)

    g = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    g[:,:,0] = input_image[:]
    g[:,:,1] = input_image[:]
    g[:,:,2] = input_image[:]
    g[:,:,3] = 255  

    border[binary==0] = 0

    b[border == 1] = (255,0,0,255)
    b[binary_border == 1] = (0,255,0,255)

    cropped_image = Util.crop_by_bbox(input_image, binary_bbox)
    cropped_binary_border = Util.crop_by_bbox(c, binary_bbox)
    cropped_combined_border = Util.crop_by_bbox(b, binary_bbox)
    cropped_border_only = Util.crop_by_bbox(border, binary_bbox)


    corrected_binary = Legacy.correct_merge(input_rhoana, label, border)
    corrected_binary_original = np.array(corrected_binary)
    result = np.array(input_rhoana)
    corrected_binary += result.max()
    corrected_binary[corrected_binary_original == 0] = 0

    result[corrected_binary != 0] = 0
    # result += corrected_binary.astype(np.uint64)
    np.add(result, corrected_binary.astype(np.uint64), out=result, casting='unsafe')
    cropped_result = Util.crop_by_bbox(corrected_binary, binary_bbox)

    g[corrected_binary_original==2] = (255,0,0,255)
    g[corrected_binary_original==1] = (0,255,0,255)
    cropped_fusion = Util.crop_by_bbox(g, binary_bbox)



    cropped_binary = Util.crop_by_bbox(f, binary_bbox)
    cropped_slice_overview = Util.crop_by_bbox(e, binary_bbox).copy()

    # e[binary_bbox[0]:binary_bbox[1], binary_bbox[2]] = (255,255,0,255)
    # e[binary_bbox[0]:binary_bbox[1], binary_bbox[3]] = (255,255,0,255)
    # e[binary_bbox[0], binary_bbox[2]:binary_bbox[3]] = (255,255,0,255)
    # e[binary_bbox[1], binary_bbox[2]:binary_bbox[3]] = (255,255,0,255)  

    sliceoverview = e    

    return cropped_image, cropped_binary_border, cropped_combined_border, cropped_border_only, cropped_result, result, sliceoverview, cropped_binary, cropped_fusion, cropped_slice_overview


  @staticmethod
  def get_split_error_image(input_image, input_rhoana, labels):

    if not isinstance(labels, list):
      labels = [labels]

    b = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    b[:,:,0] = input_image[:]
    b[:,:,1] = input_image[:]
    b[:,:,2] = input_image[:]
    b[:,:,3] = 255

    c = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    c[:,:,0] = input_image[:]
    c[:,:,1] = input_image[:]
    c[:,:,2] = input_image[:]
    c[:,:,3] = 255    

    d = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    d[:,:,0] = input_image[:]
    d[:,:,1] = input_image[:]
    d[:,:,2] = input_image[:]
    d[:,:,3] = 255    

    e = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    e[:,:,0] = input_image[:]
    e[:,:,1] = input_image[:]
    e[:,:,2] = input_image[:]
    e[:,:,3] = 255    

    f = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    f[:,:,0] = input_image[:]
    f[:,:,1] = input_image[:]
    f[:,:,2] = input_image[:]
    f[:,:,3] = 255    
    f[input_rhoana == labels[0]] = (255,0,0,255)
    f[input_rhoana == labels[1]] = (255,0,0,255)


    thresholded_rhoana = Util.view_labels(input_rhoana, labels, crop=False, return_it=True)
    
    cropped_rhoana_dilated = mh.dilate(thresholded_rhoana.astype(np.uint64))
    for dilate in range(30):
      cropped_rhoana_dilated = mh.dilate(cropped_rhoana_dilated)

    cropped_rhoana_bbox = mh.bbox(cropped_rhoana_dilated)
    binary_border = mh.labeled.borders(thresholded_rhoana.astype(np.bool))

    b[input_rhoana == labels[0]] = (255,0,0,255)
    c[mh.labeled.borders(Util.threshold(input_rhoana, labels[0])) == 1] = (255,0,0,255)
    d[binary_border == 1] = (255,0,0,255)
    if len(labels) > 1:
      b[input_rhoana == labels[1]] = (0,255,0,255)
      c[mh.labeled.borders(Util.threshold(input_rhoana, labels[1])) == 1] = (0,255,0,255)

    cropped_image = Util.crop_by_bbox(input_image, cropped_rhoana_bbox)
    cropped_labels = Util.crop_by_bbox(b, cropped_rhoana_bbox)
    cropped_borders = Util.crop_by_bbox(c, cropped_rhoana_bbox)
    cropped_binary_border = Util.crop_by_bbox(d, cropped_rhoana_bbox)



    cropped_binary_labels = Util.crop_by_bbox(f, cropped_rhoana_bbox)

    cropped_slice_overview = Util.crop_by_bbox(e, cropped_rhoana_bbox).copy()

    e[cropped_rhoana_bbox[0]:cropped_rhoana_bbox[1], cropped_rhoana_bbox[2]] = (255,255,0,255)
    e[cropped_rhoana_bbox[0]:cropped_rhoana_bbox[1], cropped_rhoana_bbox[3]] = (255,255,0,255)
    e[cropped_rhoana_bbox[0], cropped_rhoana_bbox[2]:cropped_rhoana_bbox[3]] = (255,255,0,255)
    e[cropped_rhoana_bbox[1], cropped_rhoana_bbox[2]:cropped_rhoana_bbox[3]] = (255,255,0,255)

    slice_overview = e    

    return cropped_image, cropped_labels, cropped_borders, cropped_binary_border, cropped_binary_labels, slice_overview, cropped_slice_overview




  @staticmethod
  def remove_border_mess(e):
    '''
    '''
    
    label_sizes = Util.get_histogram(e)

    if len(label_sizes) < 2:
      print 'weird'
      return e

    # we only want to keep the two largest labels
    largest1 = np.argmax(label_sizes[1:])+1
    label_sizes[largest1] = 0
    largest2 = np.argmax(label_sizes[1:])+1
    label_sizes[largest2] = 0
    for l,s in enumerate(label_sizes):
        if l == 0 or s == 0:
            # this label has zero pixels anyways or is the background
            continue
        
        # find neighbor for l
        neighbors = Util.grab_neighbors(e, l)

        if largest1 in neighbors:
            # prefer the largest
            e[e==l] = largest1
        elif largest2 in neighbors:
            e[e==l] = largest2

    return e

  @staticmethod
  def correct_merge(input_rhoana, label, border):
    
    rhoana_copy = np.array(input_rhoana, dtype=np.uint64)

    # split the label using the border
    binary = Util.threshold(input_rhoana, label).astype(np.uint64)

    border[binary==0] = 0
    binary[border==1] = 2

    binary_relabeled = Util.relabel(binary)

    # Util.view(binary_relabeled, color=True, large=True)

    binary_no_border = np.array(binary_relabeled, dtype=np.uint64)
    binary_no_border[border==1] = 0
    

    sizes = mh.labeled.labeled_size(binary_no_border)
    too_small = np.where(sizes < 200)
    labeled_small = mh.labeled.remove_regions(binary_no_border, too_small)
    labeled_small_zeros = Util.threshold(labeled_small, 0)
    labeled_small = Util.fill(labeled_small, labeled_small_zeros.astype(np.bool))
    binary_no_border = Util.frame_image(labeled_small).astype(np.uint64)     
    binary_no_border[binary==0] = 0

    corrected_binary = binary_no_border

    # now let's remove the possible border mess
    n = 0
    while corrected_binary.max() != 2 and n < 6:
      corrected_binary = Legacy.remove_border_mess(corrected_binary)
      corrected_binary = skimage.measure.label(corrected_binary)
      n += 1

    return corrected_binary

  @staticmethod
  def perform_auto_merge_correction(cnn, big_M, input_image, input_prob, input_rhoana, merge_errors, p, input_gold=None):
    
    def dojoVI(gt, seg):
      # total_vi = 0
      slice_vi = []    
      for i in range(len(gt)):
          current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
          # total_vi += current_vi
          slice_vi.append(current_vi)
      # total_vi /= 10
      return np.mean(slice_vi), np.median(slice_vi), slice_vi


    rhoanas = []

    # explicit copy
    bigM = [None]*len(big_M)
    for z in range(len(big_M)):
      bigM[z] = np.array(big_M[z])

    rhoana_after_merge_correction = np.array(input_rhoana)
    
    old_labels = []
    new_labels = []

    fixes = []

    for me in merge_errors:
        pred = me[2]
        if pred < p:
            fixes.append('yes')
            print 'fixing', pred
            z = me[0]
            label = me[1]
            border = me[3][0][1]
            a,b,c,d,e,f,g,h,i,j = Legacy.get_merge_error_image(input_image[z], rhoana_after_merge_correction[z], label, border)        
    

            new_rhoana = f
            rhoana_after_merge_correction[z] = new_rhoana

            # vi = UITools.VI(self._input_gold, rhoana_after_merge_correction)
            # print 'New global VI', vi[1]
            # if input_gold:
            rhoanas.append(dojoVI(input_gold, rhoana_after_merge_correction))    

            #
            # and remove the original label from our bigM matrix
            #
            bigM[z][label,:] = -3
            bigM[z][:,label] = -3

            # now add the two new labels
            label1 = new_rhoana.max()
            label2 = new_rhoana.max()-1
            new_m = np.zeros((bigM[z].shape[0]+2, bigM[z].shape[1]+2), dtype=bigM[z].dtype)
            new_m[:,:] = -1
            new_m[0:-2,0:-2] = bigM[z]

            # print 'adding', label1, 'to', z

            new_m = Legacy.add_new_label_to_M(cnn, new_m, input_image[z], input_prob[z], new_rhoana, label1)
            new_m = Legacy.add_new_label_to_M(cnn, new_m, input_image[z], input_prob[z], new_rhoana, label2)

            # re-propapage new_m to bigM
            bigM[z] = new_m
        else:
          fixes.append('no')

    return bigM, rhoana_after_merge_correction, fixes, rhoanas

  @staticmethod
  def perform_sim_user_merge_correction(cnn, big_M, input_image, input_prob, input_rhoana, input_gold, merge_errors):

      def dojo3VI(gt, seg):
        # total_vi = 0
        slice_vi = []    
        for i in range(len(gt)):
            current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
            # total_vi += current_vi
            slice_vi.append(current_vi)
        # total_vi /= 10
        return np.mean(slice_vi), np.median(slice_vi), slice_vi


      rhoanas = []

      # explicit copy
      bigM = [None]*len(big_M)
      for z in range(len(big_M)):
        bigM[z] = np.array(big_M[z])

      rhoana_after_merge_correction = np.array(input_rhoana)
      
      fixes = []

      for me in merge_errors:
          pred = me[2]
      
          z = me[0]
          label = me[1]
          border = me[3][0][1]
          a,b,c,d,e,f,g,h,i,j = Legacy.get_merge_error_image(input_image[z], rhoana_after_merge_correction[z], label, border)


          new_rhoana = f



          # check VI for this slice
          vi_before = Util.vi(input_gold[z], input_rhoana[z])
          vi_after = Util.vi(input_gold[z], f)

          # global vi


          if (vi_after < vi_before):

            
            # this is a good fix
            rhoana_after_merge_correction[z] = new_rhoana

            rhoanas.append(dojo3VI(input_gold, rhoana_after_merge_correction))

            #
            # and remove the original label from our bigM matrix
            #
            bigM[z][label,:] = -3
            bigM[z][:,label] = -3

            # now add the two new labels
            label1 = new_rhoana.max()
            label2 = new_rhoana.max()-1
            new_m = np.zeros((bigM[z].shape[0]+2, bigM[z].shape[1]+2), dtype=bigM[z].dtype)
            new_m[:,:] = -1
            new_m[0:-2,0:-2] = bigM[z]

            # print 'adding', label1, 'to', z, new_rhoana.shape, new_rhoana.max(), len(bigM)

            # if label1 >= new_m.shape[0]:
            #   new_m2 = np.zeros((new_m.shape[0]+2, new_m.shape[1]+2), dtype=bigM[z].dtype)
            #   new_m2[:,:] = -1
            #   new_m2[0:-2,0:-2] = new_m

            #   new_m = new_m2


            new_m = Legacy.add_new_label_to_M(cnn, new_m, input_image[z], input_prob[z], new_rhoana, label1)
            new_m = Legacy.add_new_label_to_M(cnn, new_m, input_image[z], input_prob[z], new_rhoana, label2)

            # re-propapage new_m to bigM
            bigM[z] = new_m


            fixes.append('Good')
          else:

            # rhoanas.append(dojoVI(input_gold, rhoana_after_merge_correction))
            # skipping this one
            fixes.append('Bad')
            continue            

      return bigM, rhoana_after_merge_correction, fixes, rhoanas





  @staticmethod
  def create_bigM_without_mask(cnn, volume, volume_prob, volume_segmentation, oversampling=False, verbose=False, max=100000):


    bigM = []
    global_patches = []

    if type(volume) is list:
      z_s = len(volume)
    else:
      z_s = volume.shape[0]

    t0 = time.time()
    for slice in range(z_s):

      image = volume[slice]
      prob = volume_prob[slice]
      segmentation = volume_segmentation[slice]

      
      patches = Patch.patchify(image, prob, segmentation, oversampling=oversampling, max=max, min_pixels=1)
      if verbose:
        print len(patches), 'generated in', time.time()-t0, 'seconds.'

      # return patches

      t0 = time.time()
      grouped_patches = Patch.group(patches)
      if verbose:
        print 'Grouped into', len(grouped_patches.keys()), 'patches in', time.time()-t0, 'seconds.'
      global_patches.append(patches)

      hist = Util.get_histogram(segmentation.astype(np.float))
      labels = len(hist)

      # create Matrix
      M = np.zeros((labels, labels), dtype=np.float)
      # .. and initialize with -1
      M[:,:] = -1



      for l_n in grouped_patches.keys():

        l = int(l_n.split('-')[0])
        n = int(l_n.split('-')[1])

        # test this patch group for l and n
        prediction = Patch.test_and_unify(grouped_patches[l_n], cnn)

        # fill value into matrix
        M[l,n] = prediction
        M[n,l] = prediction


      # now the matrix for this slice is filled
      bigM.append(M)

    return bigM


  @staticmethod
  def VI(gt, seg):
      # total_vi = 0
      slice_vi = []    

      if type(gt) is list:
        z_s = len(gt)
      else:
        z_s = gt.shape[0]

      for i in range(z_s):
          current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
          # total_vi += current_vi
          slice_vi.append(current_vi)
      # total_vi /= 10
      return np.mean(slice_vi), np.median(slice_vi), slice_vi


  @staticmethod
  def add_new_label_to_M(cnn, m, input_image, input_prob, input_rhoana, label1):

    # calculate neighbors of the two new labels
    label1_neighbors = Util.grab_neighbors(input_rhoana, label1)
    for l_neighbor in label1_neighbors:
      # recalculate new neighbors of l

      if l_neighbor == 0:
          # ignore neighbor zero
          continue

      prediction = Patch.grab_group_test_and_unify(cnn, input_image, input_prob, input_rhoana, label1, l_neighbor, oversampling=False)

      m[label1,l_neighbor] = prediction
      m[l_neighbor,label1] = prediction

    return m

  @staticmethod
  def splits_global_from_M_automatic(cnn, big_M, volume, volume_prob, volume_segmentation, volume_groundtruth=np.zeros((1,1)), sureness_threshold=0.95, smallest_first=False, oversampling=False, verbose=True, maxi=10000, FP=False):
    '''
    '''

    rhoanas = []
    def dojoVI(gt, seg):
      # total_vi = 0
      slice_vi = []    
      for i in range(len(gt)):
          current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
          # total_vi += current_vi
          slice_vi.append(current_vi)
      # total_vi /= 10
      return np.mean(slice_vi), np.median(slice_vi), slice_vi


    # explicit copy
    bigM = [None]*len(big_M)
    for z in range(len(big_M)):
      bigM[z] = np.array(big_M[z])
    # for development, we just need the matrix and the patches
    # return bigM, None, global_patches

    out_volume = np.array(volume_segmentation)
    # return out_volume

    good_fix_counter = 0
    bad_fix_counter = 0
    # error_rate = 0
    fixes = []
    vi_s_30mins = []

    superMax = -np.inf
    j = 0 # minute counter
    # for i in range(60): # no. corrections in 1 minute
    #for i in range(17280): # no. corrections in 24 h
    i = 0
    time_counter = 0
    while True: # no time limit
      # print 'Correction', i

      if (j>0 and j % 30 == 0):
        # compute VI every 30 minutes
        vi_after_30_min = []
        for ov in range(out_volume.shape[0]):
            vi = Util.vi(volume_groundtruth[ov], out_volume[ov])
            vi_after_30_min.append(vi)
        vi_s_30mins.append(vi_after_30_min)
        j = 0
        time_counter += 1
        print time_counter*30, 'minutes done bigM_max=', superMax

      if i>0 and i % 12 == 0:
        # every 12 corrections == 1 minute
        j += 1
        # print 'minutes', j
      i+=1


      superMax = -np.inf
      superL = -1
      superN = -1
      superSlice = -1

      #
      for slice in range(len(bigM)):
          max_in_slice = bigM[slice].max()
          largest_indices = np.where(bigM[slice]==max_in_slice)
          # print largest_indices
          if max_in_slice > superMax:
              
              # found a new large one
              l,n = largest_indices[0][0], largest_indices[1][0]
              superSlice = slice
              superL = l
              superN = n
              superMax = max_in_slice

              # print 'found', l, n, slice, max_in_slice
          
      if superMax < sureness_threshold:
        print superMax
        break



      # print 'merging', superL, superN, 'in slice ', superSlice, 'with', superMax

      image = volume[superSlice]
      prob = volume_prob[superSlice]
      # segmentation = volume_segmentation[slice]
      groundtruth = volume_groundtruth[superSlice]



      ### now we have a new max
      slice_with_max_value = np.array(out_volume[superSlice])

      rollback_slice_with_max_value = np.array(slice_with_max_value)

      # print slice_with_max_value.dtype, groundtruth.dtype

      last_vi = Util.vi(slice_with_max_value, groundtruth)

      # now we merge
      # print 'merging', superL, superN
      slice_with_max_value[slice_with_max_value == superN] = superL


    
      after_merge_vi = Util.vi(slice_with_max_value, groundtruth)
      # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
      
      # pxlsize = len(np.where(before_segmentation_copy == l)[0])
      # pxlsize2 = len(np.where(before_segmentation_copy == n)[0])


      good_fix = False
      # print '-'*80
      # print 'vi diff', last_vi-after_merge_vi
      if after_merge_vi < last_vi:
        #
        # this is a good fix
        #
        good_fix = True
        good_fix_counter += 1
        # Util.view_labels(before_segmentation_copy,[l,n], crop=False)
        # print 'good fix'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2   
        fixes.append((1, superMax))       
      else:
        #
        # this is a bad fix
        #
        good_fix = False
        bad_fix_counter += 1
        fixes.append((0, superMax))
        # print 'bad fix, excluding it..'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2

      if FP:
        # focused proofreading

        new_m = bigM[superSlice].copy()

        label1 = superL
        label2 = superN

        # grab old neighbors of label 2 which are now neighbors of label1
        label2_neighbors = Util.grab_neighbors(out_volume[superSlice], superN)
        for l_neighbor in label2_neighbors:

          if l_neighbor == 0:
            continue

          if superL == l_neighbor:
            continue

          # get old score
          old_score = new_m[superN, l_neighbor]

          label1_neighbor_score = new_m[superL, l_neighbor]
          
          # print old_score, label1_neighbor_score

          # and now choose the max of these two
          new_m[label1, l_neighbor] = max(label1_neighbor_score, old_score)
          new_m[l_neighbor, label1] = max(label1_neighbor_score, old_score)


        # label2 does not exist anymore
        new_m[:,label2] = -2
        new_m[label2, :] = -2     

        bigM[superSlice] = new_m.copy() 



      else:
        # reset all l,n entries
        bigM[superSlice][superL,:] = -2
        bigM[superSlice][:, superL] = -2
        bigM[superSlice][superN,:] = -2
        bigM[superSlice][:, superN] = -2

        # re-calculate neighbors
        # grab new neighbors of l
        l_neighbors = Util.grab_neighbors(slice_with_max_value, superL)

        for l_neighbor in l_neighbors:
          # recalculate new neighbors of l

          if l_neighbor == 0:
              # ignore neighbor zero
              continue

          prediction = Patch.grab_group_test_and_unify(cnn, image, prob, slice_with_max_value, superL, l_neighbor, oversampling=oversampling)
          # print superL, l_neighbor
          # print 'new pred', prediction
          bigM[superSlice][superL,l_neighbor] = prediction
          bigM[superSlice][l_neighbor,superL] = prediction




      out_volume[superSlice] = slice_with_max_value

      rhoanas.append(dojoVI(volume_groundtruth, out_volume))


    return bigM, out_volume, fixes, vi_s_30mins, rhoanas


  @staticmethod
  def splits_global_from_M(cnn, big_M, volume, volume_prob, volume_segmentation, volume_groundtruth=np.zeros((1,1)), hours=.5, randomize=False, error_rate=0, oversampling=False, verbose=False, FP=False):

    rhoanas = []
    def dojoVI(gt, seg):
      # total_vi = 0
      slice_vi = []    
      for i in range(len(gt)):
          current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
          # total_vi += current_vi
          slice_vi.append(current_vi)
      # total_vi /= 10
      return np.mean(slice_vi), np.median(slice_vi), slice_vi


    # explicit copy
    bigM = [None]*len(big_M)
    for z in range(len(big_M)):
      bigM[z] = np.array(big_M[z])

    # for development, we just need the matrix and the patches
    # return bigM, None, global_patches

    out_volume = np.array(volume_segmentation)
    # return out_volume

    good_fix_counter = 0
    bad_fix_counter = 0
    # error_rate = 0
    fixes = []
    vi_s_30mins = []

    superMax = -np.inf
    j = 0 # minute counter
    # for i in range(60): # no. corrections in 1 minute
    #for i in range(17280): # no. corrections in 24 h
    if hours == -1:
      # no timelimit == 30 days
      hours = 24*30

    corrections_time_limit = int(hours * 60 * 12)
    time_counter = 0
    # print 'limit',corrections_time_limit
    # for i in range(corrections_time_limit):
    i = 0
    while True: # no time limit
      # print 'Correction', i
      i+=1
      if (j>0 and j % 30 == 0):
        # compute VI every 30 minutes
        vi_after_30_min = []
        for ov in range(out_volume.shape[0]):
            vi = Util.vi(volume_groundtruth[ov], out_volume[ov])
            vi_after_30_min.append(vi)
        vi_s_30mins.append(vi_after_30_min)
        j = 0
        time_counter += 1
        if verbose:
          print time_counter*30, 'minutes done bigM_max=', superMax

      if i>0 and i % 12 == 0:
        # every 12 corrections == 1 minute
        j += 1
        # print 'minutes', j

      superMax = -np.inf
      superL = -1
      superN = -1
      superSlice = -1

      #
      for slice in range(len(bigM)):
          max_in_slice = bigM[slice].max()

          largest_indices = np.where(bigM[slice]==max_in_slice)
          # print largest_indices
          if max_in_slice > superMax:
            
              # found a new large one
              l,n = largest_indices[0][0], largest_indices[1][0]
              superSlice = slice
              superL = l
              superN = n
              superMax = max_in_slice

              # print 'found', l, n, slice, max_in_slice
          
      if superMax <= 0.:
        # print 'SUPERMAX 0'
        break

      if randomize:
        superMax = .5
        superSlice = np.random.choice(len(bigM))

        uniqueIDs = np.where(bigM[superSlice] > -3)

        superL = np.random.choice(uniqueIDs[0])

        neighbors = Util.grab_neighbors(volume_segmentation[superSlice], superL)
        superN = np.random.choice(neighbors)


      # print 'merging', superL, superN, 'in slice ', superSlice, 'with', superMax

      image = volume[superSlice]
      prob = volume_prob[superSlice]
      # segmentation = volume_segmentation[slice]
      groundtruth = volume_groundtruth[superSlice]



      ### now we have a new max
      slice_with_max_value = np.array(out_volume[superSlice])

      rollback_slice_with_max_value = np.array(slice_with_max_value)

      last_vi = Util.vi(slice_with_max_value, groundtruth)

      # now we merge
      # print 'merging', superL, superN
      slice_with_max_value[slice_with_max_value == superN] = superL


    
      after_merge_vi = Util.vi(slice_with_max_value, groundtruth)
      # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
      
      # pxlsize = len(np.where(before_segmentation_copy == l)[0])
      # pxlsize2 = len(np.where(before_segmentation_copy == n)[0])


      good_fix = False
      # print '-'*80
      # print 'vi diff', last_vi-after_merge_vi
      if after_merge_vi < last_vi:
        #
        # this is a good fix
        #
        good_fix = True
        good_fix_counter += 1
        # Util.view_labels(before_segmentation_copy,[l,n], crop=False)
        # print 'good fix'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2   
        fixes.append((1, superMax))       
      else:
        #
        # this is a bad fix
        #
        good_fix = False
        bad_fix_counter += 1
        fixes.append((0, superMax))
        # print 'bad fix, excluding it..'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2

      #
      #
      # ERROR RATE
      #
      rnd = random.random()
      # print rnd
      if rnd < error_rate:
        # no matter what, this is a user error
        good_fix = not good_fix
        # print 'user err'
      


      if FP:
        # focused proofreading
        new_m = bigM[superSlice].copy()
        bigM[superSlice][superL,superN] = -2
        bigM[superSlice][superN,superL] = -2
        

        if good_fix:

          label1 = superL
          label2 = superN


          # grab old neighbors of label 2 which are now neighbors of label1
          label2_neighbors = Util.grab_neighbors(rollback_slice_with_max_value, superN)
          for l_neighbor in label2_neighbors:

            if l_neighbor == 0:
              continue

            if superL == l_neighbor:
              continue

            # get old score
            old_score = new_m[superN, l_neighbor]

            if old_score < 0:
              continue

            label1_neighbor_score = new_m[superL, l_neighbor]
            
            if label1_neighbor_score < 0:
              continue

            print old_score, label1_neighbor_score, max(label1_neighbor_score, old_score)

            # and now choose the max of these two
            new_m[label1, l_neighbor] = max(label1_neighbor_score, old_score)
            new_m[l_neighbor, label1] = max(label1_neighbor_score, old_score)


          # label2 does not exist anymore
          new_m[:,label2] = -2
          new_m[label2, :] = -2     

          bigM[superSlice] = new_m.copy() 

          rhoanas.append(dojoVI(volume_groundtruth, out_volume))


        else:

          slice_with_max_value = rollback_slice_with_max_value
          # bigM[superSlice] = new_m.copy() 

        



      else:

        # reset all l,n entries
        bigM[superSlice][superL,superN] = -2
        bigM[superSlice][superN,superL] = -2

        if good_fix:

          bigM[superSlice][superL,:] = -2
          bigM[superSlice][:, superL] = -2
          bigM[superSlice][superN,:] = -2
          bigM[superSlice][:, superN] = -2

          # old_neighbors = Util.grab_neighbors(rollback_slice_with_max_value, superL)
          # for n in old_neighbors:
          #   neighbors = Util.grab_neighbors(slice_with_max_value, n)

          #   for k in neighbors:

          #     if bigM[superSlice][n,k] == -2:
          #       continue

          #     prediction = Patch.grab_group_test_and_unify(cnn, image, prob, slice_with_max_value, n, k, oversampling=oversampling)
          #     # print superL, l_neighbor
          #     # print 'new pred', prediction
          #     bigM[superSlice][n,k] = prediction
          #     bigM[superSlice][k,n] = prediction



          # re-calculate neighbors
          # grab new neighbors of l
          l_neighbors = Util.grab_neighbors(slice_with_max_value, superL)

          for l_neighbor in l_neighbors:
            # recalculate new neighbors of l

            if l_neighbor == 0:
                # ignore neighbor zero
                continue

            prediction = Patch.grab_group_test_and_unify(cnn, image, prob, slice_with_max_value, superL, l_neighbor, oversampling=oversampling)
            # print superL, l_neighbor
            # print 'new pred', prediction
            bigM[superSlice][superL,l_neighbor] = prediction
            bigM[superSlice][l_neighbor,superL] = prediction

          outvol2 = out_volume.copy()
          outvol2[superSlice] = slice_with_max_value

          rhoanas.append(dojoVI(volume_groundtruth, outvol2))

        else:

          slice_with_max_value = rollback_slice_with_max_value




      out_volume[superSlice] = slice_with_max_value

      
    print 'done'
    return bigM, out_volume, fixes, vi_s_30mins, rhoanas

  @staticmethod
  def plot_arand(data,filename=None):

    median_input_vi = np.median(data.values()[0])
    median_input_vi_count = len(data.values())+3

    fig, ax = plt.subplots(figsize=(22,22))
    ax.plot(range(median_input_vi_count), [median_input_vi]*median_input_vi_count, 'k:' , color='gray', linewidth=2, label='Avg. input VI')

    objects = data.keys()

    y_pos = range(1,len(objects)+1)

    # setp(r['whiskers'], color='black', lw=2) 
    plt.ylabel('Adapted Rand Error', labelpad=20)

    # plt.setp(plt.xticks()[1], rotation=45)
    ax.tick_params(axis='both', which='major', pad=15)
    plt.ylim([0.0,0.7])
    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : 48}

    plt.rc('font', **font)
    
    bp = plt.boxplot(data.values(), 0, 'gD', whis=1.5)
    plt.setp(bp['boxes'], linewidth=4)
    plt.setp(bp['medians'], linewidth=4)        
    plt.xticks(y_pos, objects)

    if filename:
      plt.savefig(filename)

    plt.show()

  @staticmethod
  def plot_vis(data,filename=None):

    median_input_vi = np.median(data.values()[0])
    median_input_vi_count = len(data.values())+3

    fig, ax = plt.subplots(figsize=(22,22))
    ax.plot(range(median_input_vi_count), [median_input_vi]*median_input_vi_count, 'k:' , color='gray', linewidth=2, label='Avg. input VI')

    objects = data.keys()

    y_pos = range(1,len(objects)+1)

    # setp(r['whiskers'], color='black', lw=2) 
    plt.ylabel('Variation of Information', labelpad=20)

    plt.setp(plt.xticks()[1], rotation=45)
    ax.tick_params(axis='both', which='major', pad=15)
    plt.ylim([0.0,0.7])
    font = {'family' : 'normal',
    #         'weight' : 'bold',
            'size'   : 26}

    plt.rc('font', **font)
    
    bp = plt.boxplot(data.values(), 0, 'gD', whis=1.5)
    plt.setp(bp['boxes'], linewidth=4)
    plt.setp(bp['medians'], linewidth=4)        
    plt.xticks(y_pos, objects)

    if filename:
      plt.savefig(filename)

    plt.show()


  @staticmethod
  def plot_vis_error_rate(data,dojo_avg_user_mean,dojo_best_user_mean,filename=None):

    # x_labels = range(10)
    x_labels = [float(l)/100. for l in range(21)]# for l in x_labels:
        
    fig, ax = plt.subplots(figsize=(22,22))
    ax.plot(x_labels, data.values(), 'o')

    ax.plot(x_labels, data.values(), 'k--', label='Guided (Simulated)', linewidth=4)

    ax.plot(x_labels, [dojo_avg_user_mean]*len(data.keys()), 'k', label='Dojo average', color='red', linewidth=4)
    ax.plot(x_labels, [dojo_best_user_mean]*len(data.keys()), 'k', label='Dojo best', color='blue', linewidth=4)
    legend = ax.legend(loc='upper right')
    ax.tick_params(axis='both', which='major', pad=15)
    plt.ylabel('Variation of Information', labelpad=20)
    plt.xlabel('User Error Rate', labelpad=20)
    plt.ylim([0.0,0.7])
    font = {'family' : 'normal',
    #         'weight' : 'bold',
            'size'   : 26}
    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)

    plt.show()

  @staticmethod
  def plot_vi_simuser(data, filename=None):
    fig, ax = plt.subplots(figsize=(22,22))

    x_marks = range(len(data))
    def green_func(x, a, b, c):
        return a*np.exp(-b*x)+c
    popt, _ = curve_fit(green_func, x_marks, data)

    xx = np.linspace(0,len(data),100)
    mediany = green_func(xx, *popt)

    # ax.plot(x_marks, median_vis_per_min, 'r', label='Simulated User')

    ax.plot(xx, mediany, linewidth=4, label='Guided (Simulated)')
    # ax.axvline(x=403, ymin=0, ymax=.245, color='b', linestyle='dashed', linewidth=2)

    plt.ylabel('Variation of Information', labelpad=20)

    plt.xlabel('Corrections', labelpad=20)
    plt.xlim([0,len(data)])
    # plt.ylim([0.4,0.5])


    legend = ax.legend(loc='upper right')

    font = {'family' : 'normal',
            'size'   : 26}

    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)

    plt.show()    



  @staticmethod
  def plot_vi_combined_no_interpolation(automatic, simuser, filename=None):

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)  



    def green_func(x, a, b, c):
        return a*np.exp(-b*x)+c

    fig, ax = plt.subplots(figsize=(22,22))

    # x_automatic = range(len(automatic))
    # popt, _ = curve_fit(green_func, x_automatic, automatic)
    xx_automatic = np.linspace(0,len(automatic),len(automatic))
    # y_automatic = green_func(xx_automatic, *popt)

    # x_simuser = range(len(simuser))    
    # popt, _ = curve_fit(green_func, x_simuser, simuser)    
    xx_simuser = np.linspace(0,len(simuser),len(simuser))
    # y_simuser = green_func(xx_simuser, *popt)


    ax.plot(xx_simuser, simuser, color=tableau20[4], linewidth=4, label='GP* (sim.)')
    ax.plot(xx_automatic, automatic, color=tableau20[6], linewidth=4, label='GP* (autom.)')
    ax.axvline(x=1972, ymin=0, ymax=.507, color=tableau20[0], linestyle='dashed', linewidth=2)
    ax.tick_params(axis='both', which='major', pad=15)
    plt.ylabel('Adapted Rand Error', labelpad=20)

    plt.xlabel('Corrections', labelpad=20)
    plt.xlim([0,len(simuser)])
    plt.ylim([0.0,0.7])


    legend = ax.legend(loc='upper right')

    font = {'family' : 'sans-serif',
            'size'   : 48}

    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)

    plt.show()        

  @staticmethod
  def plot_vi_combined(automatic, simuser, filename=None):

    def green_func(x, a, b, c):
        return a*np.exp(-b*x)+c

    fig, ax = plt.subplots(figsize=(22,22))

    x_automatic = range(len(automatic))
    popt, _ = curve_fit(green_func, x_automatic, automatic)
    xx_automatic = np.linspace(0,len(simuser),len(simuser))
    y_automatic = green_func(xx_automatic, *popt)

    x_simuser = range(len(simuser))    
    popt, _ = curve_fit(green_func, x_simuser, simuser)    
    xx_simuser = np.linspace(0,len(simuser),len(simuser))
    y_simuser = green_func(xx_simuser, *popt)


    ax.plot(xx_simuser, y_simuser, 'green', linewidth=4, label='Guided (Simulated)')
    ax.plot(xx_automatic, y_automatic, 'red', linewidth=4, label='Automatic Corrections(p=.95)')
    # ax.axvline(x=403, ymin=0, ymax=.245, color='b', linestyle='dashed', linewidth=2)

    plt.ylabel('Variation of Information', labelpad=20)

    plt.xlabel('Corrections', labelpad=20)
    plt.xlim([0,len(xx_simuser)])
    # plt.ylim([0.4,0.5])


    legend = ax.legend(loc='upper right')

    font = {'family' : 'normal',
            'size'   : 30}

    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)

    plt.show()        

  @staticmethod
  def plot_roc(roc_vals, filename=None, title=None):

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)  



    if filename:
      plt.figure(figsize=(22,22))
      font = {'family' : 'sans-serif',
              'size'   : 36}

      plt.rc('font', **font)      
      linewidth = 4
    else:
      plt.figure(figsize=(5,5))
      linewidth = 1
    for j,v in enumerate(roc_vals):
        fpr = roc_vals[v][0]
        tpr = roc_vals[v][1]    
        roc_auc = roc_vals[v][2]
        plt.plot(fpr, tpr, label=v+' [area = %0.2f]' % roc_auc, linewidth=linewidth, color=tableau20[j*2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
      plt.title(title)
    ax = plt.gca()


    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(axis='both', which='major', pad=15)
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='lower right')


    if filename:
      plt.savefig(filename)
    plt.show()

  @staticmethod
  def plot_roc_zoom(roc_vals, filename=None, title=None):


    fig = plt.figure(figsize=(22,22))
    for v in roc_vals:
        fpr = roc_vals[v][0]
        tpr = roc_vals[v][1]    
        roc_auc = roc_vals[v][2]


        plt.plot(fpr, tpr, label=v+' (area = %0.2f)' % roc_auc, linewidth=4)
        # plt.plot(xx, yy, label=v+' (area = %0.2f)' % roc_auc, linewidth=4)
    # plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.3])
    plt.ylim([0.7, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
      plt.title(title)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(axis='both', which='major', pad=15)
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='lower right')
    # plt.legend(loc="lower right")
    font = {'family' : 'normal',
            'size'   : 26}

    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)
    plt.show()

  @staticmethod
  def plot_pc(roc_vals, filename=None, title=None):

    plt.figure(figsize=(22,22))
    for v in roc_vals:
        precision = roc_vals[v][0]
        recall = roc_vals[v][1]    
        pc_auc = roc_vals[v][2]
        plt.plot(recall, precision, label=v+' (area = %0.2f)' % pc_auc, linewidth=4)
    plt.plot([1, 0], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title:
      plt.title(title)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='lower left')      
    ax.tick_params(axis='both', which='major', pad=15)
    font = {'family' : 'normal',
            'size'   : 26}

    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)
    plt.show()

  @staticmethod
  def plot_pc_zoom(roc_vals, filename=None, title=None):

    plt.figure(figsize=(22,22))
    for v in roc_vals:
        precision = roc_vals[v][0]
        recall = roc_vals[v][1]    
        pc_auc = roc_vals[v][2]
        plt.plot(recall, precision, label=v+' (area = %0.2f)' % pc_auc, linewidth=4)
    # plt.plot([1, 0], [0, 1], 'k--')
    plt.xlim([0.7, 1.0])
    plt.ylim([0.7, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if title:
      plt.title(title)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='lower left')      
    ax.tick_params(axis='both', which='major', pad=15)
    # plt.legend(loc="lower right")
    font = {'family' : 'normal',
            'size'   : 26}

    plt.rc('font', **font)

    if filename:
      plt.savefig(filename)
    plt.show()
