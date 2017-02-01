import glob
import h5py
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import partition_comparison
import os
from scipy import ndimage as nd
import skimage.measure
import sys
import tempfile
import tifffile as tif
import time
import urllib



class Util(object):

  @staticmethod
  def read_cremi_section(DATADIR, z, verbose=True):
    image = sorted(glob.glob(DATADIR+'image/*'))
    probs = sorted(glob.glob(DATADIR+'prob/*'))
    gold = sorted(glob.glob(DATADIR+'gold/*'))
    rhoana = sorted(glob.glob(DATADIR+'rhoana/*'))

    image = tif.imread(image[z])
    prob = tif.imread(probs[z])
    gold = tif.imread(gold[z])
    rhoana = mh.imread(rhoana[z])

    #convert ids from rgb to single channel
    rhoana_single = np.zeros((rhoana.shape[0], rhoana.shape[1]), dtype=np.uint64)
    rhoana_single[:, :] = rhoana[:,:,0]*256*256 + rhoana[:,:,1]*256 + rhoana[:,:,2]

    # relabel the segmentations
    gold = Util.relabel(gold)
    rhoana_single = Util.relabel(rhoana_single)

    return image, prob, gold, rhoana_single

  @staticmethod
  def read_section(path, z, verbose=True):
    '''
    '''
    image = sorted(glob.glob(os.path.join(path, 'image', '*'+str(z)+'.png')))
    mask = sorted(glob.glob(os.path.join(path, 'mask', '*'+str(z)+'.png')))   
    gold = sorted(glob.glob(os.path.join(path, 'gold', '*'+str(z)+'.png')))
    rhoana = sorted(glob.glob(os.path.join(path, 'rhoana', '*'+str(z)+'.png')))
    prob = sorted(glob.glob(os.path.join(path, 'prob', '*'+str(z)+'.tif')))

    if verbose:
      print 'Loading', os.path.basename(image[0])

    image = mh.imread(image[0])
    mask = mh.imread(mask[0]).astype(np.bool)
    gold = mh.imread(gold[0])
    rhoana = mh.imread(rhoana[0])
    prob = tif.imread(prob[0])

    #convert ids from rgb to single channel
    rhoana_single = np.zeros((rhoana.shape[0], rhoana.shape[1]), dtype=np.uint64)
    rhoana_single[:, :] = rhoana[:,:,0]*256*256 + rhoana[:,:,1]*256 + rhoana[:,:,2]
    gold_single = np.zeros((gold.shape[0], gold.shape[1]), dtype=np.uint64)
    gold_single[:, :] = gold[:,:,0]*256*256 + gold[:,:,1]*256 + gold[:,:,2]

    # relabel the segmentations
    gold_single = Util.relabel(gold_single)
    rhoana_single = Util.relabel(rhoana_single)


    #mask the rhoana output
    rhoana_single[mask==0] = 0


    return image, prob, mask, gold_single, rhoana_single

      
  @staticmethod
  def get_histogram(array):
    '''
    '''
    return mh.fullhistogram(array.astype(np.uint64))

  @staticmethod
  def get_largest_label(array, ignore_zero=False):
    '''
    '''

    hist = Util.get_histogram(array)

    if ignore_zero:
      hist[0] = 0


    return np.argmax(hist)

  @staticmethod
  def normalize_labels(array):
    '''
    '''
    return mh.labeled.relabel(array)

  @staticmethod
  def relabel(array):

    relabeled_array = np.array(array)
  
    relabeled_array = skimage.measure.label(array).astype(np.uint64)
    # relabeled_array[relabeled_array==0] = relabeled_array.max()
    
    return Util.normalize_labels(relabeled_array)[0]

  @staticmethod
  def load_colormap(f):
    '''
    '''
    hdf5_file = h5py.File(f, 'r')
    list_of_names = []
    hdf5_file.visit(list_of_names.append) 
    return hdf5_file[list_of_names[0]].value    

  @staticmethod
  def colorize(segmentation):
    '''
    '''
    if tempfile.tempdir:
      # caching only works if we have a valid temp. directory
      tempfolder = os.path.join(tempfile.tempdir, 'dojo_proofreading_study')
    cm_path = os.path.join(tempfolder, 'colorMap.hdf5')

    cm = Util.load_colormap(cm_path)
    segmentation = cm[segmentation % len(cm)]
    return segmentation

  @staticmethod
  def view(array,color=True,large=False,crop=False, text=None, no_axis=True, file=''):
    
    if large:
      figsize = (10,10)
    else:
      figsize = (3,3)

    fig = plt.figure(figsize=figsize)

    if crop:
      array = mh.croptobbox(array)

    if text:
      text = '\n\n\n'+str(text)
      fig.text(0,1,text)      

    if no_axis:
      plt.axis('off')

    if color:
      plt.imshow(Util.colorize(array), picker=True)
      iii = Util.colorize(array)
    else:
      plt.imshow(array, cmap='gray', picker=True)
      iii = array

    if file!='':
      mh.imsave(file, iii.astype(np.uint8))
      #plt.savefig(file)

  @staticmethod
  def view_rgba(patch, text=''):
    '''
    '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    text = ''+str(text)
    fig.text(0,0.5,text)

    image = patch[0]
    prob = patch[1]
    if patch.shape[0] > 2:
      binary = patch[2]
    if patch.shape[0] > 3:
      border = patch[3]

    ax1.axis('off')
    ax1.imshow(image, cmap='gray')

    ax2.axis('off')
    ax2.imshow(prob, cmap='gray')

    if patch.shape[0] > 2:
      ax3.axis('off')
      ax3.imshow(binary, cmap='gray')
      
    if patch.shape[0] > 3:
      ax4.axis('off')
      ax4.imshow(border, cmap='gray')    

  @staticmethod
  def view_labels(array, labels, crop=True, large=True, return_it=False):

    if type(labels) != type(list()):
      labels = [labels]

    out = np.zeros(array.shape)

    for l in labels:

      l_arr = Util.threshold(array, l)
      out[l_arr == 1] = out.max()+1

    if crop:
      out = mh.croptobbox(out)

    if large:
      figsize = (10,10)
    else:
      figsize = (3,3)

    if return_it:
      return out

    fig = plt.figure(figsize=figsize)      

    plt.imshow(out)


  @staticmethod
  def propagate_max_overlap(rhoana, gold):

      out = np.array(rhoana)
      
      rhoana_labels = Util.get_histogram(rhoana.astype(np.uint64))
      
      for l,k in enumerate(rhoana_labels):
          if l == 0 or k==0:
              # ignore 0 since rhoana does not have it
              continue
          values = gold[rhoana == l]
          largest_label = Util.get_largest_label(values.astype(np.uint64))
      
          out[rhoana == l] = largest_label # set the largest label from gold here
      


      return out

  @staticmethod
  def grab_neighbors(array, label):

      thresholded_array = Util.threshold(array, label)
      thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

      copy = np.array(array)
      copy[thresholded_array_dilated != thresholded_array_dilated.max()] = 0
      copy[thresholded_array == 1] = 0

      copy_hist = Util.get_histogram(copy.astype(np.uint64))

      copy_hist[0] = 0 # ignore zeros
      # copy_hist[label] = 0 # ignore ourselves
      return np.where(copy_hist>0)[0]


  @staticmethod
  def threshold(array, value):
    '''
    '''
    output_array = np.zeros(array.shape)

    output_array[array == value] = 1

    return output_array

  @staticmethod
  def frame_image(image, shape=(75,75)):
    framed = np.array(image)
    framed[:shape[0]/2+1] = 0
    framed[-shape[0]/2+1:] = 0
    framed[:,0:shape[0]/2+1] = 0
    framed[:,-shape[0]/2+1:] = 0

    return framed

  @staticmethod
  def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """    
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, 
                                    return_distances=False, 
                                    return_indices=True)
    return data[tuple(ind)]
    
  @staticmethod
  def crop_by_bbox(array, bbox, offset=0):

    return array[bbox[0]-offset:bbox[1]+offset, bbox[2]-offset:bbox[3]+offset]

  @staticmethod
  def vi(array1, array2):
    '''
    '''
    return partition_comparison.variation_of_information(array1.ravel(), array2.ravel())
        


  @staticmethod
  def load(z):
    '''
    Loads a given slice (z) and returns the image, groundtruth and 
    the initial segmentation.
    '''

    # add caching of the loaded files
    if tempfile.tempdir:
      # caching only works if we have a valid temp. directory
      tempfolder = os.path.join(tempfile.tempdir, 'dojo_proofreading_study')
      if not os.path.exists(tempfolder):
        os.mkdir(tempfolder)
    else:
      # caching will only work for the current instance
      tempfolder = tempfile.mkdtemp()


    base_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/{0}/{1}.tif?raw=true"

    data = {}

    # the coordinates of our subvolume in AC4
    x = 210
    y = 60
    #z = 50 # we dont need since we have already cut the z
    dim_x = dim_y = 400

    for what in ['image', 'groundtruth', 'segmentation', 'prob']:
      current_url = base_url.format(what, z)

      #
      current_tempfolder = os.path.join(tempfolder, what)
      if not os.path.exists(current_tempfolder):
        os.mkdir(current_tempfolder)
      current_tempfilename = os.path.join(current_tempfolder, str(z)+'.tif')

      if not os.path.exists(current_tempfilename):
        # we need to download
        tmpfile = urllib.urlretrieve(current_url, filename=current_tempfilename)[0]
      else:
        tmpfile = current_tempfilename

      data[what] = tif.imread(tmpfile)

      data[what] = data[what][y:y+dim_y,x:x+dim_x]

    #
    # also grab color map
    #
    current_tempfilename = os.path.join(tempfolder, 'colorMap.hdf5')
    if not os.path.exists(current_tempfilename):
      current_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/colorMap.hdf5?raw=true"
      tmpfile = urllib.urlretrieve(current_url, filename=current_tempfilename)[0]

    return data['image'], 255.-data['prob'], data['groundtruth'], data['segmentation']

  @staticmethod
  def load_all():
    '''
    Loads all slices and returns three volumes containing images,
    groundtruths and the initial segmentations.
    '''
    print 'Loading..'

    images = []
    probs = []
    groundtruths = []
    segmentations = []

    for z in range(10):
      image, prob, groundtruth, segmentation = Util.load(z)

      images.append(image)
      probs.append(prob)
      groundtruths.append(groundtruth)
      segmentations.append(segmentation)
        
      time.sleep(1)
      sys.stdout.write("\r%d%%" % ((z+1)*10))
      sys.stdout.flush()        

    images = np.stack(images, axis=0)
    probs = np.stack(probs, axis=0)
    groundtruths = np.stack(groundtruths, axis=0)
    segmentations = np.stack(segmentations, axis=0)

    return images, probs, groundtruths, segmentations

  @staticmethod
  def show(images, probs, golds=np.zeros((1)), segmentations=np.zeros((1))):
    '''
    '''
    z_count = 1
    if images.ndim == 3:
      z_count = images.shape[0]

    no_items = 2
    if golds.shape[0] != 1:
      no_items = 3
    if segmentations.shape[0] != 1:
      no_items = 4

    for z in range(z_count):
      fig, axis = plt.subplots(1, no_items, figsize=(no_items*3,5))

      if no_items == 4:
        text = 'Slice '+str(z)
        fig.text(0,0.5,text)

      image = images
      prob = probs
      gold = golds
      segmentation = segmentations
      if z_count > 1:
        image = images[z]
        prob = probs[z]
        gold = golds[z]
        segmentation = segmentations[z]

      axis[0].axis('off')
      axis[0].imshow(image, cmap='gray')

      axis[1].axis('off')
      if no_items == 2:
        axis[1].imshow(Util.colorize(prob))
        return
      elif no_items == 3:
        axis[1].imshow(Util.colorize(prob))
        axis[2].axis('off')
        axis[2].imshow(Util.colorize(gold))
        return

      else:
        axis[1].imshow(prob, cmap='gray')

        axis[2].axis('off')
        axis[2].imshow(Util.colorize(gold))

        axis[3].axis('off')
        axis[3].imshow(Util.colorize(segmentation))

  @staticmethod
  def show_labels(image, segmentation, labels, segmentation2=np.zeros((1)), crop=True):
    '''
    '''
    l = Util.view_labels(segmentation, labels, crop=False, large=False, return_it=True).astype(np.uint8)
    Util.show(image, l, segmentation2)

