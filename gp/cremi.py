import glob
import h5py
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import partition_comparison
import os
from scipy import ndimage as nd
import skimage.measure
import tifffile as tif



class Cremi(object):

  @staticmethod
  def read_section(path, z, verbose=True):
    '''
    '''
    image = sorted(glob.glob(os.path.join(path, 'image', '*'+str(z).zfill(9)+'_image.png')))
    gold = sorted(glob.glob(os.path.join(path, 'gold', '*'+str(z).zfill(8)+'.tif')))
    rhoana = sorted(glob.glob(os.path.join(path, 'rhoana', '*'+str(z).zfill(9)+'_neuroproof.png')))
    prob = sorted(glob.glob(os.path.join(path, 'prob', '*'+str(z).zfill(9)+'_membrane-membrane.png')))

    if verbose:
      print 'Loading', os.path.basename(image[0])

    image = mh.imread(image[0])
    # mask = mh.imread(mask[0]).astype(np.bool)
    gold = tif.imread(gold[0])
    rhoana = mh.imread(rhoana[0])
    prob = mh.imread(prob[0])

    #convert ids from rgb to single channel
    rhoana_single = np.zeros((rhoana.shape[0], rhoana.shape[1]), dtype=np.uint64)
    rhoana_single[:, :] = rhoana[:,:,0]*256*256 + rhoana[:,:,1]*256 + rhoana[:,:,2]
    # gold_single = np.zeros((gold.shape[0], gold.shape[1]), dtype=np.uint64)
    # gold_single[:, :] = gold[:,:,0]*256*256 + gold[:,:,1]*256 + gold[:,:,2]

    # relabel the segmentations
    # gold_single = Util.relabel(gold_single)
    # rhoana_single = Util.relabel(rhoana_single)


    #
    # SNIPPET FOR CONVERTING HFD5 data to single images
    #
# hdf5_file = h5py.File('/home/d/data/CREMI/sample_C_20160501.hdf')
# list_of_names = []
# hdf5_file.visit(list_of_names.append)
# data = hdf5_file['volumes/labels/neuron_ids'].value
# for z in range(data.shape[0]):
    
#     slice = data[z]
#     tif.imsave('/home/d/data/CREMI/C/gold/'+str(z).zfill(8)+'.tif', slice)


# # hdf5_file.close()


    # return image, prob, mask, gold_single, rhoana_single
    return image, prob, gold, rhoana_single
