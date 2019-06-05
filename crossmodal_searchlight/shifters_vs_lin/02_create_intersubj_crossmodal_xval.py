# copied on 2019/06/03 from /hpc/scalp/tchamina/HomoFaber/MVPY/mvpa_generictask_sylvain_sourcecode/06_intersubj_create_xval_roi_decoding_within_across_modalities.py
# adapted to do leave_one_subject_out instead of leave_two_subjects_out to prepare crossmodal searchlight

import numpy as np
import os
import os.path as op
import glob
import joblib
import pandas
import sys

import nibabel as nb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

from nilearn.image import new_img_like, concat_imgs, index_img, smooth_img, mean_img
#from nilearn.decoding import SearchLight
from nilearn.masking import intersect_masks, apply_mask

from sklearn.model_selection import LeavePOut


root_dir = '/hpc/scalp/tchamina/HomoFaber'
spm_modelname = 'Analysis_single_bayes_final_normalized'

subjects_list = ['homofaber_00', 'homofaber_01', 'homofaber_02', 'homofaber_03', 'homofaber_04',
                 'homofaber_05', 'homofaber_06', 'homofaber_07', 'homofaber_08', 'homofaber_09',
                 'homofaber_11', 'homofaber_12', 'homofaber_13', 'homofaber_14', 'homofaber_16']

mvpa_question = 'Lin_vs_Fil'

# number of subjects to be left out in the cross-validation
# (used to be 2 in the first ROI-based analyses)
# now, we use 1 for crossmodal searchlight
n_leftout_subjects = 1


mvpa_subdir = 'mvpa_{}_analyses'.format(mvpa_question)

classif_metadata_dir = op.join(root_dir,
                               mvpa_subdir,
                               'mvpa_task_definition',
                               'within_sess_perms_correct_trials')

roi_data_dir = op.join(root_dir,
                       mvpa_subdir,
                       'roi_data')

roi_xval_dir = op.join(root_dir,
                       mvpa_subdir,
                       'roi_xval_within_across_modalities')

if not(op.exists(roi_xval_dir)):
    os.makedirs(roi_xval_dir)
    print('Creating new directory to save results: {}'.format(roi_xval_dir))

roi_xval_path = op.join(roi_xval_dir,"xval_inds_leave{}subjectsout.jl".format(n_leftout_subjects))



y = []
subj_vect = []
fmri_nii_list = []
subjmask_list = []
for subj_ind, subject in enumerate(subjects_list):
    classif_metadata_path = op.join(classif_metadata_dir,subject+'_classif_metadata.jl')
    [y_subj,session_xval,modality,beta_numbers] = joblib.load(classif_metadata_path)
    y.extend(y_subj)
    subj_vect.extend(subj_ind*np.ones(len(y_subj)))

"""
mvpa_subdir = 'mvpa_{}_analyses'.format(mvpa_question)
permutations_dir = op.join(root_dir, mvpa_subdir, 'permutations', 'within_sess_perms_correct_trials')

# build group-wise permutations from subject-specific permutations
for subj_ind,subject in enumerate(subjects_list):
    permutations_path = op.join(permutations_dir,subject+'_permutations.jl')
    subj_permuts = joblib.load(permutations_path)
    if subj_ind == 0:
        allsubj_permuts = subj_permuts
    else:
        shift = allsubj_permuts.shape[1]
        allsubj_permuts = np.hstack([allsubj_permuts,shift+subj_permuts])

print(allsubj_permuts.shape)

n_permuts = allsubj_permuts.shape[0]
"""


modality_list = ['A','V']

lnso_cv = LeavePOut(n_leftout_subjects)
n_splits = lnso_cv.get_n_splits(subjects_list, subjects_list, subjects_list)

print(n_splits)

allsplits_xval_inds = []

for split_ind, (trainsubj_inds,testsubj_inds) in enumerate(lnso_cv.split(subjects_list, subjects_list, subjects_list)):
    # initialize struct for storing all train and test inds for this split
    xval_inds = dict()
    for modality in modality_list:
        xval_inds['train_{}'.format(modality)] = []
        xval_inds['test_{}'.format(modality)] = []

    shift_ind = 0

    # loop over subjects... test whether it is a train or a test subject!
    # add this data in train or test indices accordingly!
    for subj_ind, subject in enumerate(subjects_list):
        if np.sum(trainsubj_inds==subj_ind):
            # this is a train subject
            subj_nature = "train"
        else:
            subj_nature = "test"
        # read metadata classif info for this subject
        classif_metadata_path = op.join(classif_metadata_dir,subject+'_classif_metadata.jl')
        [y_subj,session_xval,modality,beta_numbers] = joblib.load(classif_metadata_path)
        # construct dummy variables to then construct our customized crossval
        sess_nbrs = np.unique(session_xval)
        sess_mods = []
        for current_sess in sess_nbrs:
            sessfilter = np.array(session_xval==current_sess)
            allstim = modality[sessfilter]
            current_sess_stim = np.unique(allstim)
            if len(current_sess_stim) > 1:
                print('Probleeeeeeeeeeeeeeem')
            else:
                sess_mods.append(current_sess_stim[0])
        sess_mods = np.array(sess_mods)
        # loop over modalities
        for modality in modality_list:
            # find session with this modality
            sess_inds = sess_nbrs[np.where(sess_mods==modality)[0]]
            # find beta indices corresponding to these sessions!
            current_filter = np.array(pandas.Series(session_xval).isin(sess_inds))
            beta_inds = np.where(current_filter)[0]
            current_string = '{}_{}'.format(subj_nature,modality)
            xval_inds[current_string].extend(beta_inds+shift_ind)
        # update shift of within subj indices to prepare next subj
        shift_ind = shift_ind+len(y_subj)
    # a bit of validation: check that the number of unique inds is ok!
    all_inds_this_split = np.concatenate([xval_inds['test_A'],xval_inds['test_V'],xval_inds['train_A'],xval_inds['train_V']])
    print(len(np.unique(all_inds_this_split)))
    # append these indices and go to next split of the xval
    allsplits_xval_inds.append(xval_inds)

# save all this!
joblib.dump([allsplits_xval_inds,np.array(y)], roi_xval_path, compress=3)







"""
lpso = LeavePGroupsOut(n_groups=2)
n_splits = lpso.get_n_splits(subj_vect, subj_vect, subj_vect)

y = np.array(y)
allpermuts_perfs = np.zeros([n_permuts,n_splits])
for perm_ind in range(n_permuts):
    y_permuted = y[allsubj_permuts[perm_ind,:]]
    single_split_path_list = []
    all_perfs = []
    print("Launching cross-validation for permutation {:04d} of {:04d}...".format(perm_ind+1,n_permuts))
    for split_ind, (train_inds,test_inds) in enumerate(lpso.split(subj_vect,subj_vect,subj_vect)):
        #print("...split {:02d} of {:02d}".format(split_ind+1, n_splits))
        single_split = [(train_inds,test_inds)]
        X_train = X[train_inds]
        X_test = X[test_inds]
        y_train = y_permuted[train_inds]
        y_test = y_permuted[test_inds]
        n_samples = len(y_train)
        class_labels = np.unique(y_train)
        n_classes = len(class_labels)
        weights_list = []
        for c in class_labels:
            weight = float(n_samples) / (n_classes * np.sum(y_train == c))
            weights_list.append(weight)
        class_weight = {class_labels[0]: weights_list[0], class_labels[1]: weights_list[1]}
        #print('Class weights used for classifier estimation:', class_weight)
        # define our estimator (which uses the weights!)
        #weighted_clf = SVC(kernel="linear", class_weight=class_weight, C=0.001)
        clf = LogisticRegression(C=0.1, class_weight=class_weight)
        #clf = Ridge()
        #clf = SpaceNetClassifier(penalty='tv-l1')
        # now we can perform decoding!
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        perf = balanced_accuracy(y_pred,y_test)
        #print(perf)
        all_perfs.append(perf)
    # save true labels results
    if perm_ind==0:
        true_res_path = op.join(roi_res_dir,"true_decoding_perf_VOI_{}_{:02d}.jl".format(region,size))
        joblib.dump(all_perfs,true_res_path,compress=3)

    print(np.mean(all_perfs))
    allpermuts_perfs[perm_ind,:] = all_perfs

permuted_res_path = op.join(roi_res_dir,"allpermuts_decoding_perf_VOI_{}_{:02d}.jl".format(region,size))

joblib.dump(allpermuts_perfs,permuted_res_path,compress=3)
"""

"""
def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print('Wrong number of arguments')
        print('Here is an example: python 05_homofaber_intersubject_roi_decoding.py left_vs_right SMA 15')
        sys.exit(2)
    else:
        mvpa_question = args[0]
        region = args[1]
        size = int(args[2])
        roi_decoding(mvpa_question,region,size)



if __name__ == "__main__":
    main()
"""
