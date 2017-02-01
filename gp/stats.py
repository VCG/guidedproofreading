import collections
import cPickle as pickle
import lasagne
import numpy as np
import os

from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion

from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_fscore_support, f1_score, precision_recall_curve, average_precision_score, zero_one_loss

import matplotlib.pyplot as plt    

import gp
from util import Util


class Stats(object):

  @staticmethod
  def create_multiplot():
    '''
    '''
    pass




  @staticmethod
  def analyze_users(FP_USERS, gold, rhoana, oracle=None, clampX=500, clampY=False, filename=None, hideXlabels=True, hideYlabels=True, DATADIR = '/home/d/GPSTUDY/', skipoutput=False, data='dojo', hline=-1, returnplt=False, showlegend=False, vilabel=False,clabel=False ):


    # def VI(gt, seg):
    #   # total_vi = 0
    #   slice_vi = []    
    #   for i in range(len(gt)):
    #       current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
    #       # total_vi += current_vi
    #       slice_vi.append(current_vi)
    #   # total_vi /= 10
    #   return np.mean(slice_vi), np.median(slice_vi), slice_vi

    # init_mean_vi = VI(gold, rhoana)[0]
    # init_median_vi = VI(gold, rhoana)[1]
    # init_vi_per_slice = VI(gold, rhoana)[2]
    # print init_median_vi
    # print 'No. users', len(FP_USERS)

    # dojo_best_vi = 0.33414926373414477
    # cylinder_best_vi = 0.27683609273291143

    # newrhoana = 'ui_results.p'
    # times = 'times.p'
    # corrections = 'corrections.p'
    # correction_vis = 'correction_vis.p'

    # fp_outputs = []
    # fp_times = []
    # fp_corrections = []
    # fp_vis =[]

    # for i,f in enumerate(FP_USERS):
    #     with open(DATADIR+FP_USERS[i]+'/'+newrhoana, 'rb') as f:
    #         fp_newrhoana = pickle.load(f)
    #         fp_outputs.append(fp_newrhoana)
    #     with open(DATADIR+FP_USERS[i]+'/'+times, 'rb') as f:
    #         fp_time = pickle.load(f)
    #         fp_time = [int(v) for v in fp_time]
    #         fp_times.append(fp_time)
    #     with open(DATADIR+FP_USERS[i]+'/'+corrections, 'rb') as f:
    #         fp_correction = pickle.load(f)       
    #         fp_corrections.append(fp_correction)
    #     with open(DATADIR+FP_USERS[i]+'/'+correction_vis, 'rb') as f:
    #         fp_correction_vis = pickle.load(f) 
    #         fp_vis.append(fp_correction_vis)

    # fp_times_mean = []
    # for t in fp_times:
    #     fp_times_mean.append(np.mean(t))
    # print 'Avg. correction time',np.mean(fp_times_mean)

    # fp_corrections_mean = []
    # fp_accepted_corrections = []

    # for c in fp_corrections:
    #     fp_corrections_mean.append(len(c))
    #     a_counter = 0
    #     for d in c:
    #         if d[1] == '1':
    #             # user accepted
    #             a_counter +=1
    #     fp_accepted_corrections.append(a_counter)
        
    # print 'Avg. corrections',np.mean(fp_corrections_mean)
    # print 'Avg. accepted', np.mean(fp_accepted_corrections)

    # fp_vi_per_c_per_user = []
    # merge_vi_per_c_per_user = []
    # split_vi_per_c_per_user = []
    # first_split_vis = []
    # last_split_vis = []
    # for k,v in enumerate(fp_corrections):
    #     vis = [init_median_vi]
    #     merge_vis = [init_median_vi]
    #     split_vis = [init_median_vi]
    #     correctionindex = 0
    #     stored_first_split_vi = False
    #     for j,c in enumerate(v):
    #         # if c[0] == 'merge':
    #         #     vi = fp_vis[k][correctionindex]
    #         #     merge_vis.append(np.median(vi))
    #         # elif c[0] == 'split':
    #         #     vi = fp_vis[k][correctionindex]
    #         #     merge_vis.append(np.median(vi))

    #         if not stored_first_split_vi and c[0] == 'split' and c[1] == '1':
    #           # store the VI for the first split
    #           vi = fp_vis[k][correctionindex]
    #           first_split_vis.append(vi)

    #           stored_first_split_vi = True

    #         if c[1] == '1':
    #             # user accepted correction, now look up the vi
    #             vi = fp_vis[k][correctionindex]
    #             correctionindex += 1    
    #             vis.append(np.median(vi))

    #             if c[0] == 'merge':
    #               merge_vis.append(vi)
    #             elif c[0] == 'split':
    #               split_vis.append(vi)
    #         else:
    #             vis.append(vis[-1])

    #             if c[0] == 'merge':
    #               merge_vis.append(vis[-1])
    #             elif c[0] == 'split':
    #               split_vis.append(vis[-1])

    # #         print fp_corrections[k][j]
    # #         if fp_corrections[k][j][1] == '1':
    # # #             print 'a'
    # #             vi_per_c = np.median(c)
    # #         else:
    # #             vi_per_c = vis[-1]
    # #         vis.append(vi_per_c)
    #     #print len(vis)
        
    #     fp_vi_per_c_per_user.append(vis)
    #     # print vis[-1]
    #     merge_vi_per_c_per_user.append(merge_vis)
    #     split_vi_per_c_per_user.append(split_vis)

    # fig = plt.figure(figsize=(10,7))
    # # fig, ax = plt.subplots()
    # # plt.ylim([0.32, 0.55])
    # # if clampX:
    # #   plt.xlim(0,clampX)

    # # plt.axhline(y=0.4763612343909136, color='gray', linestyle=':', linewidth=2, label='Initial Segmentation')
    # # plt.axhline(y=0.33414926373414477, color='gray', linestyle='--', linewidth=2, label='Best Possible')
    # plt.axhline(y=init_median_vi, color='gray', linestyle=':', linewidth=2, label='Initial Segmentation')
    # plt.axhline(y=best_vi, color='gray', linestyle='--', linewidth=2, label='Best Possible')    
    # plt.xlabel('Correction')
    # plt.ylabel('Variation of Information')
    # # legend = ax.legend(loc='upper right')

    # font = {'family' : 'sans-serif',
    #         'weight' : 'normal',
    #         'size'   : 22}

    # plt.rc('font', **font)

    # for u in fp_vi_per_c_per_user:
    #     plt.plot(u, linewidth=3)

    # if oracle:
    #   plt.plot(oracle, linewidth=2, linestyle=':')

    # if filename:
    #   plt.savefig(filename)
    # plt.show()


    # fp_vi_per_slice_per_user = []
    # fp_vi_per_slice = [0]*50
    # for o in fp_outputs:
    #     fp_vi_per_slice_per_user.append(VI(gold, o)[2])
    #     last_split_vis.append(VI(gold, o)[2])
    # for u in fp_vi_per_slice_per_user:
    #     for z,v in enumerate(u):
    #         fp_vi_per_slice[z] += v
    # for z in range(50):
    #     fp_vi_per_slice[z] /= len(fp_outputs)


    # return fp_vi_per_slice, first_split_vis, last_split_vis#, last_split_vis#merge_vi_per_c_per_user, split_vi_per_c_per_user



    def VI(gt, seg):
      # total_vi = 0
      slice_vi = []    
      for i in range(len(gt)):
          current_vi = Util.vi(gt[i].astype(np.uint64), seg[i].astype(np.uint64))
          # total_vi += current_vi
          slice_vi.append(current_vi)
      # total_vi /= 10
      return np.mean(slice_vi), np.median(slice_vi), slice_vi

    init_mean_vi = VI(gold, rhoana)[0]
    init_median_vi = VI(gold, rhoana)[1]
    init_vi_per_slice = VI(gold, rhoana)[2]

    dojo_best_vi = 0.33414926373414477
    cylinder_best_vi = 0.27683609273291143 # this is without the skipping border
    cremi_A_best_vi = 0.92974989787739659
    cremi_B_best_vi = 1.2940966349
    cremi_C_best_vi = 1.45485772678
    # cylinder_best_vi = 0.3 # this is with skipping border
    if data == 'dojo':
      best_vi = dojo_best_vi  
      clampY = [0.32, 0.55]
    elif data =='cremiA':
      best_vi = cremi_A_best_vi
      clampY = [0.85,1.1]
    elif data =='cremiB':
      best_vi = cremi_B_best_vi
      clampY = [1.2,1.8]
    elif data =='cremiC':
      best_vi = cremi_C_best_vi
      clampY = [1.35,1.96]
    else:
      best_vi = cylinder_best_vi
      clampY = [0.25, 0.5]
    if not returnplt:
      print 'No. users', len(FP_USERS)

    

    newrhoana = 'ui_results.p'
    times = 'times.p'
    corrections = 'corrections.p'
    correction_vis = 'correction_vis.p'

    fp_outputs = []
    fp_times = []
    fp_corrections = []
    fp_vis =[]

    for i,f in enumerate(FP_USERS):
        if not skipoutput:
          with open(DATADIR+FP_USERS[i]+'/'+newrhoana, 'rb') as f:
              fp_newrhoana = pickle.load(f)
              fp_outputs.append(fp_newrhoana)
        with open(DATADIR+FP_USERS[i]+'/'+times, 'rb') as f:
            fp_time = pickle.load(f)
            fp_time = [int(v) for v in fp_time]
            fp_times.append(fp_time)
        with open(DATADIR+FP_USERS[i]+'/'+corrections, 'rb') as f:
            fp_correction = pickle.load(f)       
            fp_corrections.append(fp_correction)
        with open(DATADIR+FP_USERS[i]+'/'+correction_vis, 'rb') as f:
            fp_correction_vis = pickle.load(f) 
            fp_vis.append(fp_correction_vis[1:])

    # return None, None, fp_vis

    fp_times_mean = []
    for t in fp_times:
        fp_times_mean.append(np.mean(t))
    if not returnplt:
      print 'Avg. correction time',np.mean(fp_times_mean)

    fp_corrections_mean = []
    fp_accepted_corrections = []

    for c in fp_corrections:
        fp_corrections_mean.append(len(c))
        a_counter = 0
        for d in c:
            if d[1] == '1':
                # user accepted
                a_counter +=1
        fp_accepted_corrections.append(a_counter)
    if not returnplt:    
      print 'Avg. corrections',np.mean(fp_corrections_mean)
      print 'Avg. accepted', np.mean(fp_accepted_corrections)

    fp_vi_per_c_per_user = []
    merge_vi_per_c_per_user = []
    split_vi_per_c_per_user = []
    first_split_vis = []
    last_split_vis = []
    for k,v in enumerate(fp_corrections):

        vis = [init_median_vi]
        merge_vis = [init_median_vi]
        split_vis = [init_median_vi]
        correctionindex = 0
        stored_first_split_vi = False
        for j,c in enumerate(v):
            # if c[0] == 'merge':
            #     vi = fp_vis[k][correctionindex]
            #     merge_vis.append(np.median(vi))
            # elif c[0] == 'split':
            #     vi = fp_vis[k][correctionindex]
            #     merge_vis.append(np.median(vi))

            if not stored_first_split_vi and c[0] == 'split' and c[1] == '1':
              # store the VI for the first split
              vi = fp_vis[k][correctionindex]
              first_split_vis.append(vi)

              stored_first_split_vi = True

            if c[1] == '1':
                # user accepted correction, now look up the vi
                vi = fp_vis[k][correctionindex]
                correctionindex += 1    

                vis.append(np.median(vi))


                if c[0] == 'merge':
                  merge_vis.append(vi)
                elif c[0] == 'split':
                  split_vis.append(vi)
            else:
                vis.append(vis[-1])

                if c[0] == 'merge':
                  merge_vis.append(vis[-1])
                elif c[0] == 'split':
                  split_vis.append(vis[-1])

    #         print fp_corrections[k][j]
    #         if fp_corrections[k][j][1] == '1':
    # #             print 'a'
    #             vi_per_c = np.median(c)
    #         else:
    #             vi_per_c = vis[-1]
    #         vis.append(vi_per_c)
        #print len(vis)
        
        fp_vi_per_c_per_user.append(vis)
        # print vis[-1]
        merge_vi_per_c_per_user.append(merge_vis)
        split_vi_per_c_per_user.append(split_vis)

    #return fp_vi_per_c_per_user, merge_vis, split_vis, None

    # fig = plt.figure(figsize=(10,7))
    # fig, ax = plt.subplots()
    if clampY:
        plt.ylim(clampY[0],clampY[1])
        plt.yticks(np.arange(clampY[0], clampY[1], 0.1))
    if clampX:
      plt.xlim(0,clampX)

    


    plt.axhline(y=init_median_vi, color='gray', linestyle=':', linewidth=4, label='Initial Segmentation')
    plt.axhline(y=best_vi, color='gray', linestyle='--', linewidth=4, label='Best Possible')
    if hline!=-1:
      plt.axvline(x=hline, color='red', ymin=0, ymax=.357, linewidth=4)
    if clabel:
      plt.xlabel('Correction')
    if vilabel:
      plt.ylabel('Variation of Information')
    ax = plt.gca()
    if hideXlabels:
      ax.set_xticks([])
    else:
      plt.xticks(np.arange(0, clampX+1, 1000))
    if hideYlabels:
      ax.set_yticks([])

    if showlegend:
      legend = ax.legend(loc='upper right')

    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 46}

    plt.rc('font', **font)
    plt.rc('legend',**{'fontsize':46})

    for u in fp_vi_per_c_per_user:
        plt.plot(u, linewidth=4)

    if oracle:
      plt.plot(oracle, linewidth=4, linestyle=':')

    # if filename:
    #   plt.savefig(filename)
    # plt.show()

    
    fp_vi_per_slice_per_user = []
    fp_vi_per_slice = [0]*len(gold)
    if not skipoutput:
      for o in fp_outputs:
          fp_vi_per_slice_per_user.append(VI(gold, o)[2])
          last_split_vis.append(VI(gold, o)[2])
      for u in fp_vi_per_slice_per_user:
          for z,v in enumerate(u):
              fp_vi_per_slice[z] += v
      for z in range(len(gold)):
          fp_vi_per_slice[z] /= len(fp_outputs)

    # # pass
    # # return fig

    return fp_vi_per_slice, first_split_vis, last_split_vis,fp_vis#, last_split_vis#merge_vi_per_c_per_user, split_vi_per_c_per_user


  @staticmethod
  def run_dojo_xp(cnn):

    # load dojo data
    input_image, input_prob, input_gold, input_rhoana, dojo_bbox = gp.Legacy.read_dojo_data()


    original_mean_VI, original_median_VI, original_VI_s = gp.Legacy.VI(input_gold, input_rhoana)

    # output folder for anything to store
    output_folder = '/home/d/netstatsPAPER/'+cnn.uuid+'/'
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    # find merge errors, if we did not generate them before
    merge_error_file = output_folder+'/merges_new_cnn.p'
    if os.path.exists(merge_error_file):
      print 'Loading merge errors from file..'
      with open(merge_error_file, 'rb') as f:
        merge_errors = pickle.load(f)
    else:
      print 'Finding Top 5 merge errors..'
      merge_errors = gp.Legacy.get_top5_merge_errors(cnn, input_image, input_prob, input_rhoana)
      with open(merge_error_file, 'wb') as f:
        pickle.dump(merge_errors, f)

    print len(merge_errors), ' merge errors found.'

    # we need to create a bigM for the dojo volume
    bigM_dojo_file = output_folder + '/bigM_dojo_test2.p'
    if os.path.exists(bigM_dojo_file):
      print 'Loading dojo bigM from file..'
      with open(bigM_dojo_file, 'rb') as f:
        bigM_dojo = pickle.load(f)
    else:
      print 'Creating dojo bigM..'
      bigM_dojo = gp.Legacy.create_bigM_without_mask(cnn, input_image, input_prob, input_rhoana, verbose=False)
      with open(bigM_dojo_file, 'wb') as f:
        pickle.dump(bigM_dojo, f)    



    print
    dojo_vi_95_file = output_folder + '/dojo_vi_95_t6.p'

    dojo_merge_vis = output_folder + '/dojo_merge_auto95_vis.p'
    dojo_split_vis = output_folder + '/dojo_split_auto95_vis.p'

    dojo_merge_fixes = output_folder + '/dojo_merge_auto95_fixes.p'
    dojo_split_fixes = output_folder + '/dojo_split_auto95_fixes.p'

    dojo_output_95 = output_folder + '/dojo_auto95_output.p'

    if os.path.exists(dojo_vi_95_file):
      print 'Loading merge errors p < .05 and split errors p > .95 from file..'
      with open(dojo_vi_95_file, 'rb') as f:
        dojo_vi_95 = pickle.load(f)
    else:      
      #
      # perform merge correction with p < .05
      #
      print 'Correcting merge errors with p < .05'
      bigM_dojo_05, corrected_rhoana_05, dojo_auto_merge_fixes, vi_s_per_step = gp.Legacy.perform_auto_merge_correction(cnn, bigM_dojo, input_image, input_prob, input_rhoana, merge_errors, .05, input_gold=input_gold)

      print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[0]
      print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[1]

      with open(dojo_merge_vis, 'wb') as f:
        pickle.dump(vi_s_per_step, f)


      with open(dojo_merge_fixes, 'wb') as f:
        pickle.dump(dojo_auto_merge_fixes, f) 

      #
      # perform split correction with p > .95
      #
      print 'Correcting split errors with p > .95'
      bigM_dojo_after_95, out_dojo_volume_after_auto_95, dojo_auto_fixes_95, dojo_auto_vi_s_95, vi_s_per_step2 = gp.Legacy.splits_global_from_M_automatic(cnn, bigM_dojo_05, input_image, input_prob, corrected_rhoana_05, input_gold, sureness_threshold=.95)

      dojo_vi_95 = gp.Legacy.VI(input_gold, out_dojo_volume_after_auto_95)

      with open(dojo_vi_95_file, 'wb') as f:
        pickle.dump(dojo_vi_95, f)

      with open(dojo_split_vis, 'wb') as f:
        pickle.dump(vi_s_per_step2, f)

      with open(dojo_split_fixes, 'wb') as f:
        pickle.dump(dojo_auto_fixes_95, f)       

      with open(dojo_output_95, 'wb') as f:
        pickle.dump(out_dojo_volume_after_auto_95, f) 

    print '   Mean VI improvement', original_mean_VI-dojo_vi_95[0]
    print '   Median VI improvement', original_median_VI-dojo_vi_95[1]



#
# now auto until the end
#


    print
    dojo_vi_95_file = output_folder + '/dojo_vi_till_end2.p'

    dojo_merge_vis = output_folder + '/dojo_merge_till_end_vis.p'
    dojo_split_vis = output_folder + '/dojo_split_till_end_vis.p'

    dojo_merge_fixes = output_folder + '/dojo_merge_till_end_fixes.p'
    dojo_split_fixes = output_folder + '/dojo_split_till_end_fixes.p'

    dojo_output_95 = output_folder + '/dojo_till_end_output.p'

    if os.path.exists(dojo_vi_95_file):
      print 'Loading merge errors p < .05 and split errors p > .95 from file..'
      with open(dojo_vi_95_file, 'rb') as f:
        dojo_vi_95 = pickle.load(f)
    else:      
      #
      # perform merge correction with p < .05
      #
      print 'Correcting merge errors with p < .05'
      bigM_dojo_05, corrected_rhoana_05, dojo_auto_merge_fixes, vi_s_per_step = gp.Legacy.perform_auto_merge_correction(cnn, bigM_dojo, input_image, input_prob, input_rhoana, merge_errors, .05, input_gold=input_gold)

      print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[0]
      print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[1]

      with open(dojo_merge_vis, 'wb') as f:
        pickle.dump(vi_s_per_step, f)


      with open(dojo_merge_fixes, 'wb') as f:
        pickle.dump(dojo_auto_merge_fixes, f) 

      #
      # perform split correction with p > .95
      #
      print 'Correcting split errors with p > .95'
      bigM_dojo_after_95, out_dojo_volume_after_auto_95, dojo_auto_fixes_95, dojo_auto_vi_s_95, vi_s_per_step2 = gp.Legacy.splits_global_from_M_automatic(cnn, bigM_dojo_05, input_image, input_prob, corrected_rhoana_05, input_gold, sureness_threshold=.0)

      dojo_vi_95 = gp.Legacy.VI(input_gold, out_dojo_volume_after_auto_95)

      with open(dojo_vi_95_file, 'wb') as f:
        pickle.dump(dojo_vi_95, f)

      with open(dojo_split_vis, 'wb') as f:
        pickle.dump(vi_s_per_step2, f)

      with open(dojo_split_fixes, 'wb') as f:
        pickle.dump(dojo_auto_fixes_95, f)       

      with open(dojo_output_95, 'wb') as f:
        pickle.dump(out_dojo_volume_after_auto_95, f) 

    print '   Mean VI improvement', original_mean_VI-dojo_vi_95[0]
    print '   Median VI improvement', original_median_VI-dojo_vi_95[1]






    # print
    # dojo_vi_99_file = output_folder + '/dojo_vi_99.p'
    # if os.path.exists(dojo_vi_99_file):
    #   print 'Loading merge errors p < .01 and split errors p > .99 from file..'
    #   with open(dojo_vi_99_file, 'rb') as f:
    #     dojo_vi_99 = pickle.load(f)
    # else:      
    #   #
    #   # perform merge correction with p < .01
    #   #
    #   print 'Correcting merge errors with p < .01'
    #   bigM_dojo_01, corrected_rhoana_01 = gp.Legacy.perform_auto_merge_correction(cnn, bigM_dojo, input_image, input_prob, input_rhoana, merge_errors, .01)

    #   print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_01)[0]
    #   print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_01)[1]

    #   #
    #   # perform split correction with p > .99
    #   #
    #   print 'Correcting split errors with p > .99'
    #   bigM_dojo_after_99, out_dojo_volume_after_auto_99, dojo_auto_fixes_99, dojo_auto_vi_s_99 = gp.Legacy.splits_global_from_M_automatic(cnn, bigM_dojo_01, input_image, input_prob, corrected_rhoana_01, input_gold, sureness_threshold=.99)

    #   dojo_vi_99 = gp.Legacy.VI(input_gold, out_dojo_volume_after_auto_99)

    #   with open(dojo_vi_99_file, 'wb') as f:
    #     pickle.dump(dojo_vi_99, f)

    # print '   Mean VI improvement', original_mean_VI-dojo_vi_99[0]
    # print '   Median VI improvement', original_median_VI-dojo_vi_99[1]




    print
    dojo_vi_simuser_file = output_folder + '/dojo_vi_simuser_final2.p'

    dojo_merge_vis = output_folder + '/dojo_merge_simuser_vis.p'
    dojo_split_vis = output_folder + '/dojo_split_simuser_vis.p'

    dojo_merge_fixes = output_folder + '/dojo_merge_simuser_fixes.p'
    dojo_split_fixes = output_folder + '/dojo_split_simuser_fixes.p'

    dojo_output_simuser = output_folder + '/dojo_simuser_output.p'

    if os.path.exists(dojo_vi_simuser_file):
      print 'Loading merge errors and split errors (simulated user) from file..'
      with open(dojo_vi_simuser_file, 'rb') as f:
        dojo_vi_simuser = pickle.load(f)
    else:
      #
      # perform merge correction with simulated user
      #
      print 'Correcting merge errors by simulated user (er=0)'
      bigM_dojo_simuser, corrected_rhoana_sim_user, sim_user_fixes, vi_s_per_step = gp.Legacy.perform_sim_user_merge_correction(cnn, bigM_dojo, input_image, input_prob, input_rhoana, input_gold, merge_errors)
      
      print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_sim_user)[0]    
      print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_sim_user)[1]
      
      with open(dojo_merge_vis, 'wb') as f:
        pickle.dump(vi_s_per_step, f)


      with open(dojo_merge_fixes, 'wb') as f:
        pickle.dump(sim_user_fixes, f)

      #
      # perform split correction with simulated user
      #
      print 'Correcting split errors by simulated user (er=0)'
      bigM_dojo_after, out_dojo_volume_after_sim_user, dojo_sim_user_fixes, dojo_sim_user_vi_s, vi_s_per_step2 = gp.Legacy.splits_global_from_M(cnn, bigM_dojo_simuser, input_image, input_prob, corrected_rhoana_sim_user, input_gold, hours=.5)

      dojo_vi_simuser = gp.Legacy.VI(input_gold, out_dojo_volume_after_sim_user)

      with open(dojo_vi_simuser_file, 'wb') as f:
        pickle.dump(dojo_vi_simuser, f) 

      with open(dojo_split_vis, 'wb') as f:
        pickle.dump(vi_s_per_step2, f)

      with open(dojo_split_fixes, 'wb') as f:
        pickle.dump(dojo_sim_user_fixes, f)

      with open(dojo_output_simuser, 'wb') as f:
        pickle.dump(out_dojo_volume_after_sim_user, f)

    print '   Mean VI improvement', original_mean_VI-dojo_vi_simuser[0]
    print '   Median VI improvement', original_median_VI-dojo_vi_simuser[1]

    
    # print
    # dojo_vi_simuser_er_file = output_folder + '/dojo_vi_simuser_er.p'
    # if os.path.exists(dojo_vi_simuser_er_file):
    #   print 'Loading merge errors and split errors (simulated user) with error rates from file..'
    #   with open(dojo_vi_simuser_er_file, 'rb') as f:
    #     dojo_vi_simuser_er = pickle.load(f)
    # else:
    #   #
    #   # Simulated user split correction with error rate
    #   #
    #   print '   Re-running simulated user with er=0 .. 0.2'
    #   dojo_vi_simuser_er = collections.OrderedDict()
    #   for error_rate in range(0,21):
    #     error_rate /= 100.
    #     # print '---',error_rate
    #     bigM_dojo_after, out_dojo_volume_after_sim_user, dojo_sim_user_fixes, dojo_sim_user_vi_s = gp.Legacy.splits_global_from_M(cnn, bigM_dojo, input_image, input_prob, input_rhoana, input_gold, hours=.5, error_rate=error_rate)

    #     # append the mean VI
    #     dojo_vi_simuser_er[str(error_rate)] = gp.Legacy.VI(input_gold, out_dojo_volume_after_sim_user)[1]

    #   with open(dojo_vi_simuser_er_file, 'wb') as f:
    #     pickle.dump(dojo_vi_simuser_er, f)



    #
    #
    # DOJO VIs
    #
    #
    dojo_input_vi = gp.Legacy.VI(input_gold, input_rhoana)[2]
    dojo_best_user = [0.3764043166,
                      0.3516472472,
                      0.4079547444,
                      0.4530306854,
                      0.489459557,
                      0.4783714198,
                      0.4691797846,
                      0.4852945057,
                      0.4989719721,
                      0.4631116968]

    dojo_avg_user = [0.4731860794,
                     0.4412143846,
                     0.4645102603,
                     0.4790327986,
                     0.5483534853,
                     0.5209529753,
                     0.5614397773,
                     0.5669964498,
                     0.6037881064,
                     0.5986637472]

    dojo_novice = [0.37012190195707095,
                   0.38968960153287835,
                   0.37045672764672943,
                   0.38191441070762,
                   0.45717155397457265,
                   0.4307223374738305,
                   0.46325236818430504,
                   0.5049116191382206,
                   0.45915778345523783,
                   0.5901800985629162]

    # josh
    dojo_expert1 = [0.37484603520770676,
                    0.3939621266824016,
                    0.3896524948878737,
                    0.39639562518511084,
                    0.4477210348104004,
                    0.4647934798574145,
                    0.4647357412387576,
                    0.4583758825458508,
                    0.42396064070850503,
                    0.4060052118497355]

    # alyssa
    dojo_expert2 = [0.36955775659747747,
                    0.39250293829836735,
                    0.3688303634072678,
                    0.37744240803449625,
                    0.40022644067826807,
                    0.3815527838331203,
                    0.4472774009966649,
                    0.44162415508056707,
                    0.4729849772418282,
                    0.4966401210922369]

    random_recommendations = [0.4595423295850365,
                              0.4308474043605237,
                              0.3986397439416596,
                              0.39881363108964063,
                              0.4774106038339303,
                              0.47744847314733896,
                              0.5331778295505849,
                              0.5657477626512799,
                              0.6298385736857668,
                              0.5383586601809531]


    # data = collections.OrderedDict()
    # data['Automatic\nSegmentation'] = dojo_input_vi
    # data['Dojo\n(avg. user)'] = dojo_avg_user
    # data['Dojo\n(best user)'] = dojo_best_user
    # data['Novice    \nUser'] = dojo_novice
    # data['Expert     \nUser 1'] = dojo_expert1
    # data['Expert     \nUser 2'] = dojo_expert2
    # data['Simulated   \nUser   '] = dojo_vi_simuser[2]
    # data['Random\nRecommen-\ndations'] = random_recommendations
    # data['Automatic\nCorrections\n(p=.95)'] = dojo_vi_95[2]
    # data['Automatic\nCorrections\n(p=.99)'] = dojo_vi_99[2]


    # gp.Legacy.plot_vis(data, output_folder+'/dojo_vi.pdf')

    # #
    # # simple VI plot
    # #
    # data = collections.OrderedDict()
    # data['Initial\nSegmentation'] = dojo_input_vi
    # data['Automatic\nCorrections'] = dojo_vi_95[2]
    # data['Dojo       '] = dojo_best_user
    # data['Guided    \n(Novice)    '] = dojo_novice
    # expert_sum = []
    # for i,d in enumerate(dojo_expert1):
    #   expert_sum.append((dojo_expert1[i]+dojo_expert2[i])/2.)
    # data['Guided    \n(Expert)   '] = expert_sum
    # data['Guided\n(Simulated)'] = dojo_vi_simuser[2]
    # gp.Legacy.plot_vis(data, output_folder+'/dojo_vi_simple3.pdf')

    # #
    # # simple VI plot
    # #
    # data = collections.OrderedDict()
    # data['Initial\nSegmentation'] = dojo_input_vi
    # data['Automatic\nCorrections'] = dojo_vi_95[2]
    # data['Dojo       '] = dojo_best_user
    # expert_sum = []
    # for i,d in enumerate(dojo_expert1):
    #   expert_sum.append((dojo_expert1[i]+dojo_expert2[i])/2.)
    # # data['Guided    \n(Novice)   '] = dojo_novice
    # data['Guided     '] = expert_sum
    # # data['Guided\n(Simulated)'] = dojo_vi_simuser[2]
    # gp.Legacy.plot_vis(data, output_folder+'/dojo_vi_simple2.pdf')




    # gp.Legacy.plot_vis_error_rate(dojo_vi_simuser_er, np.median(dojo_avg_user), np.median(dojo_best_user), output_folder+'/dojo_errorrate.pdf')


  @staticmethod
  def run_cylinder_xp(cnn):

    # load cylinder data
    input_image = []
    input_prob = []
    input_rhoana = []
    input_gold = []
    for z in range(250, 300):
        image, prob, mask, gold, rhoana = gp.Util.read_section('/home/d/data/cylinderNEW/', z, verbose=False)

        input_image.append(image)
        input_prob.append(255.-prob)
        input_rhoana.append(rhoana)
        input_gold.append(gold)


    original_mean_VI, original_median_VI, original_VI_s = gp.Legacy.VI(input_gold, input_rhoana)

    print 'Original median VI', original_median_VI

    # output folder for anything to store
    output_folder = '/home/d/netstatsPAPER/'+cnn.uuid+'/'
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)






    ### SKIPPING MERGE FOR NOW
    # find merge errors, if we did not generate them before
    merge_error_file = output_folder+'/merge_errors_cylinder.p'
    if os.path.exists(merge_error_file):
      print 'Loading merge errors from file..'
      with open(merge_error_file, 'rb') as f:
        merge_errors = pickle.load(f)
    else:
      print 'Finding Top 5 merge errors..'
      merge_errors = gp.Legacy.get_top5_merge_errors(cnn, input_image, input_prob, input_rhoana)
      with open(merge_error_file, 'wb') as f:
        pickle.dump(merge_errors, f)

    print len(merge_errors), ' merge errors found.'
    ####

    # we need to create a bigM for the cylinder volume
    bigM_cylinder_file = output_folder + '/bigM_cylinder.p'
    if os.path.exists(bigM_cylinder_file):
      print 'Loading cylinder bigM from file..'
      with open(bigM_cylinder_file, 'rb') as f:
        bigM_cylinder = pickle.load(f)
    else:
      print 'Creating cylinder bigM..'
      bigM_cylinder = gp.Legacy.create_bigM_without_mask(cnn, input_image, input_prob, input_rhoana, verbose=True, max=1000000)
      with open(bigM_cylinder_file, 'wb') as f:
        pickle.dump(bigM_cylinder, f)    




    print
    cylinder_vi_95_file = output_folder + '/cylinder_vi_95_w_merge.p'
    # cylinder_vi_auto_95_fixes_file = output_folder + '/cylinder_vi_95_fixes.p'
    # cylinder_auto_vis_95_file = output_folder + '/cylinder_auto_vis_95.p'


    dojo_merge_vis = output_folder + '/cylinder_merge_auto95_vis.p'
    dojo_split_vis = output_folder + '/cylinder_split_auto95_vis.p'

    dojo_merge_fixes = output_folder + '/cylinder_merge_auto95_fixes.p'
    dojo_split_fixes = output_folder + '/cylinder_split_auto95_fixes.p'

    dojo_output_95 = output_folder + '/cylinder_auto95_output.p'

    if os.path.exists(cylinder_vi_95_file):
      print 'Loading merge errors p < .05 and split errors p > .95 from file..'
      with open(cylinder_vi_95_file, 'rb') as f:
        cylinder_vi_95 = pickle.load(f)
      # with open(cylinder_auto_vis_95_file, 'rb') as f:
      #   cylinder_auto_vi_s_95 = pickle.load(f)
      # with open(cylinder_vi_auto_95_fixes_file, 'rb') as f:
      #   cylinder_auto_fixes_95 = pickle.load(f)
    else:      
      #
      # perform merge correction with p < .05
      #
      print 'Correcting merge errors with p < .05'
      bigM_cylinder_05, corrected_rhoana_05, cylinder_auto_merge_fixes, vi_s_per_step = gp.Legacy.perform_auto_merge_correction(cnn, bigM_cylinder, input_image, input_prob, input_rhoana, merge_errors, .05, input_gold=input_gold)

      print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[0]
      print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[1]

      with open(dojo_merge_vis, 'wb') as f:
        pickle.dump(vi_s_per_step, f)


      with open(dojo_merge_fixes, 'wb') as f:
        pickle.dump(cylinder_auto_merge_fixes, f) 


      #
      # perform split correction with p > .95
      #
      print 'Correcting split errors with p > .95'
      # bigM_cylinder_05 = bigM_cylinder
      # corrected_rhoana_05 = input_rhoana
      bigM_cylinder_after_95, out_cylinder_volume_after_auto_95, cylinder_auto_fixes_95, cylinder_auto_vi_s_95, vi_s_per_step2 = gp.Legacy.splits_global_from_M_automatic(cnn, bigM_cylinder_05, input_image, input_prob, corrected_rhoana_05, input_gold, sureness_threshold=.95)

      cylinder_vi_95 = gp.Legacy.VI(input_gold, out_cylinder_volume_after_auto_95)

      with open(cylinder_vi_95_file, 'wb') as f:
        pickle.dump(cylinder_vi_95, f)

      # with open(cylinder_vi_auto_95_fixes_file, 'wb') as f:
      #   pickle.dump(cylinder_auto_fixes_95, f)

      # with open(cylinder_auto_vis_95_file, 'wb') as f:
      #   pickle.dump(cylinder_auto_vi_s_95, f)        

      with open(dojo_split_vis, 'wb') as f:
        pickle.dump(vi_s_per_step2, f)

      with open(dojo_split_fixes, 'wb') as f:
        pickle.dump(cylinder_auto_fixes_95, f)       

      with open(dojo_output_95, 'wb') as f:
        pickle.dump(out_cylinder_volume_after_auto_95, f) 


    print '   Mean VI improvement', original_mean_VI-cylinder_vi_95[0]
    print '   Median VI improvement', original_median_VI-cylinder_vi_95[1]







    # print
    # cylinder_vi_0_file = output_folder + '/cylinder_vi_0.p'
    # cylinder_vi_auto_0_fixes_file = output_folder + '/cylinder_vi_0_fixes.p'
    # cylinder_auto_vis_0_file = output_folder + '/cylinder_auto_vis_0.p'    
    # if os.path.exists(cylinder_vi_0_file):
    #   print 'Loading split errors p >= .0 from file..'
    #   with open(cylinder_vi_0_file, 'rb') as f:
    #     cylinder_vi_0 = pickle.load(f)
    #   with open(cylinder_vi_auto_0_fixes_file, 'rb') as f:
    #     cylinder_auto_fixes_00 = pickle.load(f)
    #   with open(cylinder_auto_vis_0_file, 'rb') as f:
    #     cylinder_auto_vi_s_00 = pickle.load(f)


    # else:      
    #   # #
    #   # # perform merge correction with p < .01
    #   # #
    #   # print 'Correcting merge errors with p < .01'
    #   # bigM_dojo_05, corrected_rhoana_05 = gp.Legacy.perform_auto_merge_correction(cnn, bigM_dojo, input_image, input_prob, input_rhoana, merge_errors, .05)

    #   # print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[0]
    #   # print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_05)[1]

    #   #
    #   # perform split correction with p > .99
    #   #
    #   print 'Correcting split errors with p >= .0'
    #   bigM_cylinder_00 = bigM_cylinder
    #   corrected_rhoana_00 = input_rhoana
    #   bigM_cylinder_after_00, out_cylinder_volume_after_auto_00, cylinder_auto_fixes_00, cylinder_auto_vi_s_00 = gp.Legacy.splits_global_from_M_automatic(cnn, bigM_cylinder_00, input_image, input_prob, corrected_rhoana_00, input_gold, sureness_threshold=.0)

    #   cylinder_vi_0 = gp.Legacy.VI(input_gold, out_cylinder_volume_after_auto_00)

    #   with open(cylinder_vi_0_file, 'wb') as f:
    #     pickle.dump(cylinder_vi_0, f)

    #   with open(cylinder_vi_auto_0_fixes_file, 'wb') as f:
    #     pickle.dump(cylinder_auto_fixes_00, f)

    #   with open(cylinder_auto_vis_0_file, 'wb') as f:
    #     pickle.dump(cylinder_auto_vi_s_00, f)      

    # print '   Mean VI improvement', original_mean_VI-cylinder_vi_0[0]
    # print '   Median VI improvement', original_median_VI-cylinder_vi_0[1]



    print
    cylinder_vi_simuser_file = output_folder + '/cylinder_vi_simuser_final.p'
    # cylinder_fixes_simuser_file = output_folder + '/cylinder_fixes_simuser.p'
    # cylinder_vis_simuser_file = output_folder + '/cylinder_vi_s_simuser.p'


    dojo_merge_vis = output_folder + '/cylinder_merge_simuser_vis.p'
    dojo_split_vis = output_folder + '/cylinder_split_simuser_vis.p'

    dojo_merge_fixes = output_folder + '/cylinder_merge_simuser_fixes.p'
    dojo_split_fixes = output_folder + '/cylinder_split_simuser_fixes.p'

    dojo_output_simuser = output_folder + '/cylinder_simuser_output.p'    
    if os.path.exists(cylinder_vi_simuser_file):
      print 'Loading merge errors and split errors (simulated user) from file..'
      with open(cylinder_vi_simuser_file, 'rb') as f:
        cylinder_vi_simuser = pickle.load(f)
      # with open(cylinder_fixes_simuser_file, 'rb') as f:
      #   cylinder_sim_user_fixes = pickle.load(f)
      # with open(cylinder_vis_simuser_file, 'rb') as f:
      #   cylinder_sim_user_vi_s = pickle.load(f)


    else:
      # #
      # # perform merge correction with simulated user
      # #
      print 'Correcting merge errors by simulated user (er=0)'
      bigM_cylinder_simuser, corrected_rhoana_sim_user, sim_user_fixes, vi_s_per_step = gp.Legacy.perform_sim_user_merge_correction(cnn, bigM_cylinder, input_image, input_prob, input_rhoana, input_gold, merge_errors)
      
      print '   Mean VI improvement', original_mean_VI-gp.Legacy.VI(input_gold, corrected_rhoana_sim_user)[0]    
      print '   Median VI improvement', original_median_VI-gp.Legacy.VI(input_gold, corrected_rhoana_sim_user)[1]
      
      with open(dojo_merge_vis, 'wb') as f:
        pickle.dump(vi_s_per_step, f)


      with open(dojo_merge_fixes, 'wb') as f:
        pickle.dump(sim_user_fixes, f)

      #
      # perform split correction with simulated user
      #
      print 'Correcting split errors by simulated user (er=0)'
      # bigM_cylinder_simuser = bigM_cylinder
      # corrected_rhoana_sim_user = input_rhoana
      bigM_cylinder_after, out_cylinder_volume_after_sim_user, cylinder_sim_user_fixes, cylinder_sim_user_vi_s, vi_s_per_step2 = gp.Legacy.splits_global_from_M(cnn, bigM_cylinder_simuser, input_image, input_prob, corrected_rhoana_sim_user, input_gold, hours=-1)

      cylinder_vi_simuser = gp.Legacy.VI(input_gold, out_cylinder_volume_after_sim_user)

      with open(cylinder_vi_simuser_file, 'wb') as f:
        pickle.dump(cylinder_vi_simuser, f)      

      # with open(cylinder_vis_simuser_file, 'wb') as f:
      #   pickle.dump(cylinder_sim_user_vi_s, f)

      # with open(cylinder_fixes_simuser_file, 'wb') as f:
      #   pickle.dump(cylinder_sim_user_fixes, f)
      with open(dojo_split_vis, 'wb') as f:
        pickle.dump(vi_s_per_step2, f)

      with open(dojo_split_fixes, 'wb') as f:
        pickle.dump(cylinder_sim_user_fixes, f)

      with open(dojo_output_simuser, 'wb') as f:
        pickle.dump(out_cylinder_volume_after_sim_user, f)


    print '   Mean VI improvement', original_mean_VI-cylinder_vi_simuser[0]
    print '   Median VI improvement', original_median_VI-cylinder_vi_simuser[1]


    # data = collections.OrderedDict()
    # data['Automatic\nSegmentation'] = gp.Legacy.VI(input_gold, input_rhoana)[2]
    # data['Simulated   \nUser   '] = cylinder_vi_simuser[2]
    # data['Automatic\nCorrections\n(p=.95)'] = cylinder_vi_95[2]
    # data['Automatic\nCorrections\n(p=.99)'] = cylinder_vi_99[2]


    # gp.Legacy.plot_vis(data, output_folder+'/cylinder_vi.pdf')

    # #
    # # Simple VI boxplot
    # #
    # data = collections.OrderedDict()
    # data['Initial\nSegmentation'] = gp.Legacy.VI(input_gold, input_rhoana)[2]
    # data['Automatic\nCorrections'] = cylinder_vi_95[2]    
    # data['Guided\n(Simulated)'] = cylinder_vi_simuser[2]

    # gp.Legacy.plot_vis(data, output_folder+'/cylinder_vi_simple.pdf')




    # proofread_vis = [original_VI_s] + cylinder_sim_user_vi_s
    # vi_s_per_correction = [np.median(proofread_vis[0])]
    # for m in proofread_vis[1:]:
    #     for i in range(30*12):
    #         vi_s_per_correction.append(np.median(m))

    # gp.Legacy.plot_vi_simuser(vi_s_per_correction, output_folder+'/cylinder_simuser_vi.pdf')

    # proofread_vis_auto = [original_VI_s] + cylinder_auto_vi_s_00
    # vi_s_per_correction_auto = [np.median(proofread_vis_auto[0])]
    # for m in proofread_vis_auto[1:]:
    #     for i in range(30*12):
    #         vi_s_per_correction_auto.append(np.median(m))

    # # gp.Legacy.plot_vi_combined(vi_s_per_correction_auto, vi_s_per_correction, output_folder+'/cylinder_combined_vi.pdf')

    # gp.Legacy.plot_vi_combined_no_interpolation(vi_s_per_correction_auto, vi_s_per_correction, output_folder+'/cylinder_combined_vi_no_interpolation.pdf', sweetspot=len(cylinder_auto_fixes_95))


    # data = {}
    # y_text_fixes = [v[0] for v in cylinder_sim_user_fixes]
    # y_pred_fixes = [v[1] for v in cylinder_sim_user_fixes]
    # fpr, tpr, _ = roc_curve(y_text_fixes, y_pred_fixes)
    # roc_auc = auc(fpr, tpr)    
    # data['Cylinder Fixes'] = (fpr, tpr, roc_auc)
    # gp.Legacy.plot_roc(data, output_folder+'/cylinder_roc.pdf')

