import numpy as np
import os
import os.path as op
import joblib

import nibabel as nb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

from nilearn.image import new_img_like, concat_imgs
from nilearn.decoding import SearchLight
from nilearn.masking import intersect_masks



def balanced_accuracy(y1,y2):
    return recall_score(y1, y2, pos_label=None, average='macro')

root_dir = '/hpc/scalp/tchamina/HomoFaber'
spm_modelname = 'Analysis_single_bayes_final_normalized'

subjects_list = ['homofaber_00', 'homofaber_01', 'homofaber_02', 'homofaber_03', 'homofaber_04',
                 'homofaber_05', 'homofaber_06', 'homofaber_07', 'homofaber_08', 'homofaber_09',
                 'homofaber_11', 'homofaber_12', 'homofaber_13', 'homofaber_14', 'homofaber_16']

mvpa_question = 'Lin_vs_Fil'

n_jobs = 8


def run_crossmodal_searchlight(split_ind, train_modality, test_modality, searchlight_radius):
    """Execute a single-split searchlight analysis on homofaber experiment, with the possibility
    to do it in a crossmodal setting (train on one modality, test on either the same one or a
    different one). This saves a single-fold accuracy map.

    Parameters
    ----------
    split_ind : int
        number of the cross-validation fold
    train_modality : string
        either 'A' or 'V' (for audio or video respectively)
    test_modality : string
        either 'A' or 'V' (for audio or video respectively)
    searchlight_radius: float
        radius, in milimmeter, of the spherical searchlight region

    Returns
    -------

    """



    split_ind = 12
    train_modality = 'A'
    test_modality = 'V'
    searchlight_radius = 3





    # defining all input directories
    mvpa_subdir = 'mvpa_{}_analyses'.format(mvpa_question)
    permutations_dir = op.join(root_dir,
                               mvpa_subdir,
                               'permutations',
                               'within_sess_perms_correct_trials')
    classif_metadata_dir = op.join(root_dir,
                                   mvpa_subdir,
                                   'mvpa_task_definition',
                                   'within_sess_perms_correct_trials')

    roi_xval_dir = op.join(root_dir,
                           mvpa_subdir,
                           'roi_xval_within_across_modalities')


    # defining all output directories
    ##### OUTPUT DIRS to be DEFINED properly
    searchlight_res_dir = op.join(root_dir,
                                  mvpa_subdir,
                                  'intersubj_wCbeta_crossmodal_searchlight_res')
    if not(op.exists(searchlight_res_dir)):
        os.makedirs(searchlight_res_dir)
        print('Creating new directory to save results: {}'.format(searchlight_res_dir))
    single_split_res_dir = op.join(searchlight_res_dir,'single_split_maps')
    if not(op.exists(single_split_res_dir)):
        os.makedirs(single_split_res_dir)
        print('Creating new directory to save single split searchlight maps: {}'.format(single_split_res_dir))


    # number of subjects to be left out in the cross-validation
    # (used to be 2 in the first ROI-based analyses)
    # now, we use 1 for crossmodal searchlight
    n_leftout_subjects = 1
    roi_xval_path = op.join(roi_xval_dir,"xval_inds_leave{}subjectsout.jl".format(n_leftout_subjects))


    # read mask_nii and fmri_img if it exists already
    y = []
    subj_vect = []
    fmri_nii_list = []
    subjmask_list = []
    for subj_ind, subject in enumerate(subjects_list):
        classif_metadata_path = op.join(classif_metadata_dir,subject+'_classif_metadata.jl')
        [y_subj,session_xval,modality,beta_numbers] = joblib.load(classif_metadata_path)
        y.extend(y_subj)
        subj_vect.extend(subj_ind*np.ones(len(y_subj)))

        subject_dir = op.join(root_dir, subject)
        betas_dir = op.join(subject_dir, spm_modelname)

        for current_beta_ind in beta_numbers:
            beta_path = op.join(betas_dir, 'Cbeta_{:04d}.nii'.format(current_beta_ind))
            print(beta_path)
            beta_nii = nb.load(beta_path)
            #sbeta_nii = smooth_img(beta_nii, fwhm=fwhm)
            #sbeta_data = sbeta_nii.get_data()
            fmri_nii_list.append(beta_nii)

        subjmask_name = op.join(betas_dir,'mask.nii')
        subjmask_nii = nb.load(subjmask_name)
        subjmask_list.append(subjmask_nii)

    subj_vect = np.array(subj_vect)

    print("Intersecting the masks from all the subjects...")
    mask_nii = intersect_masks(subjmask_list)
    print("Concatenating the data from all the subjects...")
    fmri_img = concat_imgs(fmri_nii_list)

    # write mask_nii and fmri_img if it does not exist yet!


    # 0. definition of the parameters
    split_ind = 12
    train_modality = 'A'
    test_modality = 'V'
    searchlight_radius = 3

    # 1. read pre-defined xval inds
    [allsplits_xval_inds, y] = joblib.load(roi_xval_path)

    # 2. extract indices for current_split
    n_splits = len(allsplits_xval_inds)
    currentsplit_xval_inds = allsplits_xval_inds[split_ind]
    train_key = 'train_{}'.format(train_modality)
    train_inds = currentsplit_xval_inds[train_key]
    test_key = 'train_{}'.format(test_modality)
    test_inds = currentsplit_xval_inds[test_key]

    # 3. define fmri_nii_specific and y_specific for the given pair of (train_modality, test_modality)
    # need to remap fmri_nii and y to restrict them in order to minimize mapping time at the beginning
    # of searchlight fit (this, in fact, is optional, to be coded only if it represents a
    # computational bottleneck)
    # I NEED TO VALIDATE THIS BY RUNNING ONE SEARCHLIGHT WITH AND WITHOUT, coz it's dangerous ;)
    #testtrain_inds = np.concatenate([train_inds,test_inds])
    #y_specific = y[testtrain_inds]

    # 4. define single split xval and searchlight
    single_split = [(train_inds, test_inds)]
    weighted_clf = LogisticRegression(C=0.1, class_weight='balanced')
    # now we can call the searchlight with all these options
    print("...preparing searchlight for this split")
    searchlight = SearchLight(mask_nii,
                              process_mask_img=mask_nii,
                              radius=searchlight_radius,
                              n_jobs=n_jobs,
                              verbose=1,
                              cv=single_split,
                              scoring=make_scorer(balanced_accuracy),
                              estimator=weighted_clf)

    # 5. fit searchlight
    print("...fitting searchlight for this split!")
    searchlight.fit(fmri_img, y)
    #searchlight.fit(fmri_img_specific, y_specific)

    # 6. save res, with the 4 parameters in the filename

    single_split_nii = new_img_like(mask_nii,searchlight.scores_)
    single_split_path = op.join(single_split_res_dir,'intersubj_balancedacc_rad{:05.2f}mm_train{}test{}_split{:1d}of{:1d}.nii.gz'.format(searchlight_radius,train_modality,test_modality,split_ind+1,n_splits))
    print('Saving score map for {}, fold number {:02d} of {:02d}'.format(subjects_list[split_ind],split_ind+1,n_splits))
    single_split_nii.to_filename(single_split_path)
    #single_split_path_list.append(single_split_path)

def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print('Wrong number of arguments')
        print('Here is an example: python 03_homofaber_intersubject_searchlight_crossmodalities_decoding.py 3 A V 5.6')
        sys.exit(2)
    else:
        split_ind = int(args[0])
        train_modality = args[1]
        test_modality = args[2]
        searchlight_radius = float(args[3])

    run_crossmodal_searchlight(split_ind, train_modality, test_modality, searchlight_radius)

if __name__ == "__main__":
    main()





