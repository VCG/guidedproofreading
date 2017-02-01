from scipy.spatial import distance
from util import Util
import mahotas as mh
import numpy as np
import skimage.measure
import scipy.misc
import uuid
import matplotlib.pyplot as plt
import os
import random
import time

# from uglify import Uglify

class Patch(object):



  @staticmethod
  def grab(image, prob, segmentation, l, n, sample_rate=10, mode='', oversampling=False, patch_size=(75,75)):



    # grab border between l and n
    border = mh.labeled.border(segmentation, l, n)
    
    # also grab binary mask for l
    binary_l = Util.threshold(segmentation, l)
    # .. and n
    binary_n = Util.threshold(segmentation, n)

    # analyze both borders
    patches_l = Patch.analyze_border(image, prob, binary_l, binary_n, border, l, n, sample_rate=sample_rate, patch_size=patch_size, oversampling=oversampling, mode=mode)
    patches_n = Patch.analyze_border(image, prob, binary_n, binary_l, border, n, l, sample_rate=sample_rate, patch_size=patch_size, oversampling=oversampling, mode=mode)

    return patches_l, patches_n

  @staticmethod
  def group(patches):
    '''
    groups patches by label pair
    '''

    patch_grouper = {}
    for p in patches:
        # create key

        minlabel = min(p['l'], p['n'])
        maxlabel = max(p['l'], p['n'])
        key = str(minlabel)+'-'+str(maxlabel)

        if not key in patch_grouper:
            patch_grouper[key] = []

        patch_grouper[key] += [p]

    return patch_grouper


  @staticmethod
  def test_and_unify(patches, cnn, method='weighted_arithmetic_mean'):
    '''
    '''

    if method != 'weighted_arithmetic_mean':

      # just the standard mean
      return np.mean(Patch.test(patches, cnn, stats=False))

    else:

      # weighted arithmetic mean based on border length

      weights = []
      predictions = []

      for p in patches:

          # calculate the border length based on the patch size
          bbox = p['bbox']
          valid_border_points = 0
          for c in p['border']:
              if c[0] >= bbox[0] and c[0] <= bbox[1]:
                  if c[1] >= bbox[2] and c[1] <= bbox[3]:
                      # valid border point
                      valid_border_points += 1

          # pred = cnn.test_patch(p)

          if cnn.uuid.endswith('B'):

            border_prefix = ''
            if cnn.uuid.find('LB') != -1:
              border_prefix = 'larger_'

            rgba_patch = np.zeros((1,4,75,75), dtype=np.float32)
            rgba_patch[0][0] = p['image'].astype(np.float32)
            rgba_patch[0][1] = p['prob'].astype(np.float32)
            rgba_patch[0][2] = p['merged_array'].astype(np.float32)
            rgba_patch[0][3] = p[border_prefix+'border_overlap'].astype(np.float32)

            inputs = rgba_patch

          elif cnn.uuid.startswith('IPM'):

            rgb_patch = np.zeros((1,3,75,75), dtype=np.float32)
            rgb_patch[0][0] = p['image'].astype(np.float32)
            rgb_patch[0][1] = p['prob'].astype(np.float32)
            rgb_patch[0][2] = p['merged_array'].astype(np.float32)
            inputs = rgb_patch

          else:
            inputs = {}
            for i,n in enumerate(cnn.input_names):
              inputs[n] = p[cnn.input_values[i]].reshape(-1,1,75,75)


          # print inputs
          

          # normalize input_image and input_prob
          # inputs['prob_input'] = inputs['prob_input'].astype(np.float) / 255.        
          # inputs['image_input'] = inputs['image_input'].astype(np.float) / 255.
          

          pred = cnn.predict_proba(inputs)[0][1]

          # pred = cnn.predict_proba({'image_input': p['image'].reshape(-1,1,75,75),
          #  'prob_input': p['prob'].reshape(-1,1,75,75),
          #  'binary_input': p['merged_array'].reshape(-1,1,75,75),
          #  'border_input': p['border_overlap'].reshape(-1,1,75,75)})[0][1]
          weights.append(valid_border_points)
          predictions.append(pred)

      p_sum = 0
      w_sum = 0
      for i,w in enumerate(weights):
          # weighted arithmetic mean
          p_sum += w*predictions[i]
          w_sum += w

      p_sum /= w_sum

      return p_sum

  @staticmethod
  def grab_group_test_and_unify(cnn, image, prob, segmentation, l, n, sample_rate=10, oversampling=False):
    '''
    '''
    patches = []
    patches_l, patches_n = Patch.grab(image, prob, segmentation, l, n, oversampling=oversampling)
    patches += patches_l
    patches += patches_n

    grouped_patches = Patch.group(patches)

    if len(grouped_patches.keys()) != 1:
      # out of bound condition due to probability map
      return -1

    prediction = Patch.test_and_unify(grouped_patches.items()[0][1], cnn)

    return prediction

  @staticmethod
  def grab_group_test_and_unify_for_active_label(cnn, image, prob, segmentation, l, n, sample_rate=10, oversampling=False):
    '''
    '''
    patches = []
    patches_l, patches_n = Patch.grab(image, prob, segmentation, l, n, oversampling=oversampling)
    patches += patches_l
    patches += patches_n

    grouped_patches = Patch.group(patches)

    if len(grouped_patches.keys()) != 1:
      # out of bound condition due to probability map
      return -1

    prediction = Patch.test_and_unify(grouped_patches.items()[0][1], cnn)

    return feature_map

  @staticmethod
  def patchify(image, prob, segmentation, sample_rate=10, min_pixels=100, max=1000, oversampling=False, ignore_zero_neighbor=True):
    '''
    '''
    patches = []
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = len(hist)
    # print 'labels', labels
    batch_count = 0
    

    for l in range(labels):

      if l == 0:
        continue

      if hist[l] < min_pixels:

        continue

      neighbors = Util.grab_neighbors(segmentation, l)

      for n in neighbors:

        if ignore_zero_neighbor and n == 0:
          continue

        if hist[n] < min_pixels:
          continue

        
          

        # print 'grabbing', l, n
        p_l, p_n = Patch.grab(image, prob, segmentation, l, n, sample_rate, oversampling=oversampling)

        # if l == 5 and n ==6:
        #   print p_l, p_n

        patches += p_l
        patches += p_n

        if len(patches) >= max:

          return patches[0:max]        


    return patches


  @staticmethod
  def patchify_maxoverlap(image, prob, mask, segmentation, gold, sample_rate=3, min_pixels=10, oversampling=True,
                      ignore_zero_neighbor=True, patch_size=(75,75), clamp_result=True):
    '''
    '''
    no_mask=False
    # grab the mask border
    if mask.max() == 0.:
      # ignore borders
      print 'no mask'
      no_mask = True
    else:
      mask_borders = mh.labeled.borders(mask)
    
    # fill segmentation using max overlap and relabel it
    fixed = Util.propagate_max_overlap(segmentation, gold)
    fixed = Util.relabel(fixed)

    # now remove parts which are not in the fixed segmentation
    segmentation[fixed==0] = 0
    
    # grab borders of segmentation and fixed
    segmentation_borders = mh.labeled.borders(segmentation)
    fixed_borders = mh.labeled.borders(fixed)
    if not no_mask:
      fixed_borders[mask_borders == 1] = 0
    
    
    bad_borders = np.array(segmentation_borders)
    bad_borders[fixed_borders == 1] = 0
    if not no_mask:
      bad_borders[mask_borders == 1] = 0




    patches = []
    error_patches = []
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = range(len(hist))
    np.random.shuffle(labels)


    batch_count = 0
    # return segmentation_borders, bad_borders

    for l in labels:

      if l == 0:
        continue

      if hist[l] < min_pixels:
        continue

      neighbors = Util.grab_neighbors(segmentation, l)

      for n in neighbors:

        if ignore_zero_neighbor and n == 0:
          continue

        if hist[n] < min_pixels:
          continue


        # grab the border between l and n
        # t0 = time.time()
        l_n_border = mh.labeled.border(segmentation, l, n)
        # print t0-time.time()

        # check if the border is a valid split or a split error
        split_error = False
        



        # # print 'grabbing', l, n
        p_l, p_n = Patch.grab(image, prob, segmentation, l, n, sample_rate, oversampling=oversampling, patch_size=patch_size)

        for p in p_l:

          patch_center = p['border_center']
          if bad_borders[patch_center[0], patch_center[1]] == 1:
            error_patches.append(p)
          elif fixed_borders[patch_center[0], patch_center[1]] == 1:
            patches.append(p)


        for p in p_n:

          patch_center = p['border_center']
          if bad_borders[patch_center[0], patch_center[1]] == 1:
            error_patches.append(p)
          elif fixed_borders[patch_center[0], patch_center[1]] == 1:
            patches.append(p)            

        # patches += p_l
        # patches += p_n

        # if len(patches) >= max:

        #   return patches[0:max]    

    # print 'Found', len(patches), len(error_patches)

    if clamp_result:
      # print 'clamping'
      patches = patches[0:len(error_patches)]




    # we will always have less error patches

    # missing_error_patches = len(patches)-len(error_patches)

    # while missing_error_patches > 2000:

    #   new_error_patches = Patch.split_and_patchify(image, prob, fixed, 
    #                                                 min_pixels=1000,
    #                                                 max=missing_error_patches,
    #                                                 n=10, sample_rate=10, oversampling=True,
    #                                                 patch_size=(31,31))

    #   error_patches += new_error_patches
    #   missing_error_patches = len(patches)-len(error_patches)

    return error_patches, patches



  @staticmethod
  def analyze_border(image,
                     prob,
                     binary_mask,
                     binary_mask2,
                     border,
                     l=1,
                     n=0,
                     mode='',
                     patch_size=(75,75),
                     sample_rate=10,
                     oversampling=False):

      patches = []

      patch_centers = []
      border_yx = indices = zip(*np.where(border==1))

      if len(border_yx) < 2:
        # somehow border detection did not work
        # print 'no border', l,n
        return patches
      # print image.shape
      # plt.imshow(image)
      # print border_yx, l, n
      # always sample the middle point
      border_center = (border_yx[len(border_yx)/(2)][0], border_yx[len(border_yx)/(2)][1])
      # print 'a'
      patch_centers.append(border_center)


      if sample_rate > 1 or sample_rate == -1:
          if sample_rate > len(border_yx) or sample_rate==-1:
              samples = 1
          else:
              samples = len(border_yx) / sample_rate

          for i,s in enumerate(border_yx):
              
              if i % samples == 0:

                  sample_point = s

                  if not oversampling:
            
                    if distance.euclidean(patch_centers[-1],sample_point) < patch_size[0]:
                      # sample to close
                      # print 'sample to close', l, n
                      continue

                  patch_centers.append(sample_point)
          
      # borders_w_center = np.array(borders.astype(np.uint8))

      # for i,c in enumerate(patch_centers):
          


      #     borders_w_center[c[0],c[1]] = 10*(i+1)
      #     # print 'marking', c, borders_w_center.shape

      # # if len(patch_centers) > 1:
      # #   print 'PC', patch_centers

      # if l==5 and n==6:
      #   print 'HERE'
          
      for i,c in enumerate(patch_centers):

          # print 'pc',c
          
  #         for border_center in patch_centers:

          # check if border_center is too close to the 4 edges
          new_border_center = [c[0], c[1]]

          if new_border_center[0] < patch_size[0]/2:
              # return None
              continue
          if new_border_center[0]+patch_size[0]/2 >= border.shape[0]:
              # return None
              continue
          if new_border_center[1] < patch_size[1]/2:
              # return None
              continue
          if new_border_center[1]+patch_size[1]/2 >= border.shape[1]:
              # return None
              continue
          # print new_border_center, patch_size[0]/2, border_center[0] < patch_size[0]/2

          # continue


          bbox = [new_border_center[0]-patch_size[0]/2, 
                  new_border_center[0]+patch_size[0]/2,
                  new_border_center[1]-patch_size[1]/2, 
                  new_border_center[1]+patch_size[1]/2]

          ### workaround to not sample white border of probability map
          # if skip_boundaries:
          # if l==5 and n==6:
          #   print 'HERE2', bbox
          # if bbox[0] <= 33:
          #     # return None
          #     continue
          # if bbox[1] >= border.shape[0]-33:
          #     # return None
          #     continue
          # if bbox[2] <= 33:
          #     # return None
          #     continue
          # if bbox[3] >= border.shape[1]-33:
          #     # return None
          #     continue

          
          relabeled_cutout_binary_mask = skimage.measure.label(binary_mask[bbox[0]:bbox[1] + 1,
                                                                           bbox[2]:bbox[3] + 1])
          relabeled_cutout_binary_mask += 1
          center_cutout = relabeled_cutout_binary_mask[patch_size[0]/2,patch_size[1]/2]
          relabeled_cutout_binary_mask[relabeled_cutout_binary_mask != center_cutout] = 0

          relabeled_cutout_binary_mask = relabeled_cutout_binary_mask.astype(np.bool)

          cutout_border = mh.labeled.border(relabeled_cutout_binary_mask, 1, 0)

          merged_array = binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] + binary_mask2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          merged_array_border = mh.labeled.border(merged_array, 1, 0)


          # cutout_border = np.array(border[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool))
          for d in range(1):
            cutout_border = mh.dilate(cutout_border)

          for d in range(1):
            merged_array_border = mh.dilate(merged_array_border)



          isolated_border = cutout_border - merged_array_border
          isolated_border[cutout_border==0] = 0          

          larger_isolated_border = np.array(isolated_border)  
          for d in range(5):
            larger_isolated_border = mh.dilate(larger_isolated_border)

          # relabeled_cutout_border = cutout_border

          # relabeled_cutout_border = skimage.measure.label(cutout_border)
          # relabeled_cutout_border += 1 # avoid 0
          # center_cutout = relabeled_cutout_border[patch_size[0]/2,patch_size[1]/2]
          # relabeled_cutout_border[relabeled_cutout_border != center_cutout] = 0


          # # threshold for label1
          # array1 = Util.threshold(segmentation, l).astype(np.uint8)
          # threshold for label2
          # array2 = Util.threshold(segmentation, n).astype(np.uint8)
          # merged_array = array1 + array2


          

          # # # dilate for overlap
          # dilated_array1 = np.array(binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1])
          # dilated_array2 = np.array(binary_mask2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1])
          # for o in range(patch_size[0]/2):
          #     dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
          #     dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))
          # overlap = np.logical_and(dilated_array1, dilated_array2)
          # overlap[merged_array == 0] = 0

          # overlap_labeled = skimage.measure.label(overlap)
          # overlap_value = overlap_labeled[37,37]
          # print overlap_value
          # overlap_thresholded = np.zeros(overlap.shape)
          # overlap_thresholded[overlap_labeled == overlap_value] = 1
          # overlap = overlap_thresholded

          # dyn_obj = np.zeros(merged_array.shape)
          # r = 10
          # midpoint = [patch_size[0]/2, patch_size[1]/2]
          # dyn_obj[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r] = merged_array[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r]

          # dyn_bnd = np.zeros(overlap.shape)
          # r = 25
          # midpoint = [patch_size[0]/2, patch_size[1]/2]
          # dyn_bnd[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r] = overlap[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r]

          # def recreate_binaries(binary, merged_binary):
          #     # restore both labels
          #     binary = np.array(binary.reshape(75,75), dtype=np.uint8)
          #     merged_array = np.array(merged_binary.reshape(75,75), dtype=np.uint8)
              
          #     label1 = np.array(binary+merged_array, dtype=np.uint8)
          #     label1[label1 != 2] = 0
          #     label1 = label1.astype(np.bool)

          #     label2 = np.array(merged_array - label1, dtype=np.bool)
              
          #     return label1, label2

          # def recreate_dyn_bnd(label1, label2, r=10):
          #     amount = r/2
              
          #     dilated_label1 = np.array(label1)
          #     for i in range(amount):
          #         dilated_label1 = mh.dilate(dilated_label1)

          #     dilated_label2 = np.array(label2)
          #     for i in range(amount):
          #         dilated_label2 = mh.dilate(dilated_label2)
                  
          #     dyn_bnd = np.logical_and(dilated_label1, dilated_label2)
              
          #     return dyn_bnd.astype(np.bool)
                  
          # def recreate_dyn_obj(label1, label2, r=4, patch_size=(75,75)):
          #     merged_array = label1 + label2
          #     midpoint = [patch_size[0]/2, patch_size[1]/2]
              
          #     dyn_obj = np.zeros(merged_array.shape, dtype=np.bool)
          #     dyn_obj[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r] = merged_array[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r]
              
          #     return dyn_obj.astype(np.bool)
              
          # label1, label2 = recreate_binaries(relabeled_cutout_binary_mask.astype(np.bool), merged_array.astype(np.bool))


          # dyn_bnd = recreate_dyn_bnd(label1, label2)
          # dyn_obj = recreate_dyn_obj(label1, label2)

          # dyn_bnd_dyn_obj = dyn_bnd+dyn_obj

          output = {}
          # output['id'] = str(uuid.uuid4())
          output['image'] = image[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.float32) / 255.

          output['prob'] = prob[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.float32) / 255
          # output['binary1'] = binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          # output['binary'] = label1#binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
          # output['binary1'] = label1#binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
          # output['binary2'] = label2#binary_mask2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
          output['merged_array'] = merged_array.astype(np.bool)
          # output['dyn_obj'] = dyn_obj.astype(np.bool)
          # output['dyn_bnd'] = dyn_bnd.astype(np.bool)
          # output['dyn_bnd_dyn_obj'] = dyn_bnd_dyn_obj.astype(np.bool)
          output['bbox'] = bbox
          output['border'] = border_yx
          output['border_center'] = new_border_center
          output['border_overlap'] = isolated_border.astype(np.bool)
          # output['overlap'] = overlap.astype(np.bool)
          output['larger_border_overlap'] = larger_isolated_border.astype(np.bool)
          output['l'] = l
          output['n'] = n
          # output['overlap'] = overlap[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)

          # output['borders_labeled'] = borders_labeled[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]
          # output['borders_w_center'] = borders_w_center[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]

          # sanity check for empty patches
          # if output['image'].max() == 0.:
            # print 'ZEEEEROS HERE'
            # continue

          patches.append(output)
          

      return patches


  @staticmethod
  def load(PATCH_PATH, patch_size=(75,75), verbose=True):

      PATCH_PATH = os.path.expanduser('~/patches/') + os.sep + PATCH_PATH + os.sep

      t0 = time.time()

      training = np.load(PATCH_PATH+'train.npz', mmap_mode='r')
      training_targets = np.load(PATCH_PATH+'train_targets.npz')

      test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
      test_targets = np.load(PATCH_PATH+'test_targets.npz')

      if verbose:
        print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

      return training, training_targets['targets'].astype(np.uint8), test, test_targets['targets'].astype(np.uint8)

  @staticmethod
  def load_rgba(PATCH_PATH, border_prefix='border', patch_size=(75,75), verbose=True):

    PATCH_PATH_ = '/tmp/' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/pfister_lab/haehn/patches/' + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/regal/pfister_lab/haehn' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches_local/') + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches/') + os.sep + PATCH_PATH + os.sep

    PATCH_PATH = PATCH_PATH_

    t0 = time.time()

    training = np.load(PATCH_PATH+'train.npz', mmap_mode='r')
    training_targets = np.load(PATCH_PATH+'train_targets.npz')

    test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
    test_targets = np.load(PATCH_PATH+'test_targets.npz')

    if verbose:
      print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

    return training['rgba'], training_targets['targets'].astype(np.uint8), test['rgba'], test_targets['targets'].astype(np.uint8)

  @staticmethod
  def load_rgb(PATCH_PATH, patch_size=(75,75), verbose=True):

    PATCH_PATH_ = '/tmp/' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/pfister_lab/haehn/patches/' + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/regal/pfister_lab/haehn' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches_local/') + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches/') + os.sep + PATCH_PATH + os.sep

    PATCH_PATH = PATCH_PATH_

    t0 = time.time()

    training = np.load(PATCH_PATH+'train.npz', mmap_mode='r')
    training_targets = np.load(PATCH_PATH+'train_targets.npz')

    test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
    test_targets = np.load(PATCH_PATH+'test_targets.npz')

    if verbose:
      print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

    return training['rgb'], training_targets['targets'].astype(np.uint8), test['rgb'], test_targets['targets'].astype(np.uint8)

  @staticmethod
  def load_small(patch_size=(75,75), verbose=True):


    PATCH_PATH = '/n/regal/pfister_lab/haehn/FINAL/IPMLB/'

    t0 = time.time()

    training = np.load(PATCH_PATH+'train.npz', mmap_mode='r')
    training_targets = np.load(PATCH_PATH+'train_targets.npz')

    test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
    test_targets = np.load(PATCH_PATH+'test_targets.npz')

    if verbose:
      print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

    return training['rgba'], training_targets['targets'].astype(np.uint8), test['rgba'], test_targets['targets'].astype(np.uint8)


  @staticmethod
  def load_cremi(patch_size=(75,75), verbose=True):


    PATCH_PATH = '/n/regal/pfister_lab/haehn/CREMIBIG/'

    t0 = time.time()

    training = np.load(PATCH_PATH+'train.npy', mmap_mode='r')
    training_targets = np.load(PATCH_PATH+'train_targets.npy')

    test = np.load(PATCH_PATH+'test.npz.npy', mmap_mode='r')
    test_targets = np.load(PATCH_PATH+'test_targets.npz.npy')

    if verbose:
      print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

    return training, training_targets.astype(np.uint8), test, test_targets.astype(np.uint8)

  # @staticmethod
  # def load_cremi(patch_size=(75,75), verbose=True):


  #   PATCH_PATH = '/n/regal/pfister_lab/haehn/FINAL/IPMLB_before_NP/'

  #   t0 = time.time()

  #   training = np.load(PATCH_PATH+'train.npy', mmap_mode='r')
  #   training_targets = np.load(PATCH_PATH+'train_targets.npy')

  #   test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
  #   test_targets = np.load(PATCH_PATH+'test_targets.npz')

  #   if verbose:
  #     print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

  #   return training, training_targets.astype(np.uint8), test['rgba'], test_targets['targets'].astype(np.uint8)



  @staticmethod
  def load_rgba_test_only(PATCH_PATH, verbose=True):

    PATCH_PATH_ = '/tmp/' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/pfister_lab/haehn/patches/' + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/regal/pfister_lab/haehn' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches_local/') + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches/') + os.sep + PATCH_PATH + os.sep

    PATCH_PATH = PATCH_PATH_

    t0 = time.time()

    test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
    test_targets = np.load(PATCH_PATH+'test_targets.npz')

    if verbose:
      print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

    return test['rgba'], test_targets['targets'].astype(np.uint8)


  @staticmethod
  def load_rgb_test_only(PATCH_PATH, verbose=True):

    PATCH_PATH_ = '/tmp/' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/pfister_lab/haehn/patches/' + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = '/n/regal/pfister_lab/haehn' + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches_local/') + os.sep + PATCH_PATH + os.sep
    if not os.path.exists(PATCH_PATH_):
      PATCH_PATH_ = os.path.expanduser('~/patches/') + os.sep + PATCH_PATH + os.sep

    PATCH_PATH = PATCH_PATH_

    t0 = time.time()

    test = np.load(PATCH_PATH+'test.npz', mmap_mode='r')
    test_targets = np.load(PATCH_PATH+'test_targets.npz')

    if verbose:
      print 'Loaded', PATCH_PATH, 'in', time.time()-t0, 'seconds.'

    return test['rgb'], test_targets['targets'].astype(np.uint8)
