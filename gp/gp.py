import cPickle as pickle

from legacy import Legacy
from patch import Patch
from util import Util

class GP(object):
  '''
  '''

  def __init__(self, model='mouse'):
    '''
    Initialize the Guided Proofreading classifier.

    By default model='mouse' but can also be model='fruitfly'
    '''

    # we load a trained model, by default for mouse brain
    if model == 'mouse':
      cnn_path = 'models/gp_mouse.p'
    elif model == 'fruitfly':
      cnn_path = 'models/gp_fruitfly.p'
    elif model == 'human':
      cnn_path = 'models/gp_human.p'
      print 'You freak!!!'
    else:
      raise Error('Model ' + str(model) + ' not found.')

    with open(cnn_path, 'rb') as f:
      self._cnn = pickle.load(f)
      self._cnn.uuid = 'IPMLB'

    print 'Loaded GP model for', model, 'brain.'

  
  def rank(self, image, prob, segmentation, label1, label2):
    '''
    Rank the intersection between label1 and label2 for split errors.

    This is the same as an affinity score.

    Returns
      1 If label1 and label2 should be merged.
      ..
      0 If label1 and label2 should NOT be merged.

        or
      
      -1 If there was a problem.
    '''
    return Patch.grab_group_test_and_unify(self._cnn,
                                           image,
                                           prob,
                                           segmentation,
                                           label1, 
                                           label2)


  def find_merge_error(self, image, prob, segmentation, label):
    '''
    Find a merge error in a segment.

    This generates 50 borders within the segment and ranks them
    using the CNN.

    Returns
      The corrected segment.
        and
      Rank (1-p) for the found merge error.
        1 If confident
        ..
        0 If uncertain


        or

      -1 If there was a problem.
    '''

    # create binary mask
    binary = Util.threshold(segmentation, label)

    # here potential boundaries are generated and ranked by the CNN
    results = Legacy.fix_single_merge(self._cnn,
                                      image,
                                      prob,
                                      binary,
                                      N=50, 
                                      erode=True, 
                                      invert=True,
                                      dilate=True,
                                      border_seeds=True,
                                      oversampling=False)

    if len(results) > 0:
      
      # sort the prediction/border tuples (prediction, border)-tupels
      sorted_pred = sorted(results, key=lambda x: x[0])
      lowest_rank = sorted_pred[0][0]
      border = sorted_pred[0][1]

      corrected = Legacy.correct_merge(segmentation, label, border)

      return corrected, 1.-lowest_rank

    else:

      return -1
