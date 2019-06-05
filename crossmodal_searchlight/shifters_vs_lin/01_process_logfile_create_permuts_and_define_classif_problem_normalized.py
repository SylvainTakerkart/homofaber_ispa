# copied on 2019/06/03 from /hpc/scalp/tchamina/HomoFaber/MVPY/mvpa_Shifters_vs_Lin_ThC_sourcecode/01_process_logfile_create_permuts_and_define_classif_problem_normalized_THC_writable.py
# nothing changed (we had fixed a bug with Thierry a few weeks before

import numpy as np

import os
import os.path as op
import glob
import pandas
import joblib


n_permuts = 5000

subjects_list = ['homofaber_00', 'homofaber_01', 'homofaber_02', 'homofaber_03', 'homofaber_04',
                 'homofaber_05', 'homofaber_06', 'homofaber_07', 'homofaber_08', 'homofaber_09',
                 'homofaber_11', 'homofaber_12', 'homofaber_13', 'homofaber_14', 'homofaber_16']

root_dir = '/hpc/scalp/tchamina/HomoFaber'

logfiles_subdir = 'Final_logs'

mvpa_question = 'Shifters_vs_Lin'

mvpa_subdir = 'mvpa_{}_analyses'.format(mvpa_question)

permutations_dir = op.join(root_dir, mvpa_subdir, 'permutations', 'within_sess_perms_correct_trials')
classif_metadata_dir = op.join(root_dir, mvpa_subdir, 'mvpa_task_definition', 'within_sess_perms_correct_trials')

if not(op.exists(permutations_dir)):
    os.makedirs(permutations_dir)
    print('Creating new directory to save permutations: {}'.format(permutations_dir))
if not(op.exists(classif_metadata_dir)):
    os.makedirs(classif_metadata_dir)
    print('Creating new directory to save the definition of the classification task: {}'.format(classif_metadata_dir))


for subject in subjects_list:
    print('Processing subject {}'.format(subject))

    logfile_path = op.join(root_dir,logfiles_subdir,subject+'.txt')

    df = pandas.read_csv(logfile_path,sep='\t', dtype='object')

    # filter to keep only correct responses
    filter = np.array(df['CORR'].isin(['1']))
    df = df[filter]

    # filter to keep only images corresponding to the classes of interest
    column_name = 'type'
    classes = [['Beg','End'],['Lin']]
    class_labels = ['Shifters', 'Lin']
    #column_name = 'form'
    #classes = [['A'], ['B'], ['C']]
    #class_labels = ['A', 'B', 'C']
    allclasses = [element for sublist in classes for element in sublist]
    filter = np.array(df[column_name].isin(allclasses)) # problem is here!
    print(np.unique(df['sess']))
    df = df[filter]
    # now, create the permuted indices, session per session
    # create also the true y, the vector containing the numbers of corresponding
    # beta maps, and two vectors containing the session number and the modality
    y = []
    session = []
    modality = []
    beta_numbers = []
    session_labels = np.unique(df['sess'])

    print(session_labels)
    shift_size = 0
    for sess_ind, current_sess_label in enumerate(session_labels):
        filter = np.array(df['sess'].isin([str(current_sess_label)]))
        session_effective_size = np.sum(filter)
        current_sess_permuts = []
        # add first permutation with real labels!
        current_sess_permuts.append(np.arange(session_effective_size)+shift_size)
        # create permutations
        for perm_ind in range(n_permuts-1):
            current_sess_permuts.append(np.random.permutation(session_effective_size)+shift_size)
        if sess_ind == 0:
            all_permuts = np.array(current_sess_permuts)
        else:
            all_permuts = np.hstack([all_permuts,np.array(current_sess_permuts)])
        shift_size = shift_size + session_effective_size
        # fill in all other vectors / list
        session.extend(df['sess'][filter])
        modality.extend(df['stim'][filter])
        beta_numbers.extend(pandas.to_numeric(df['beta'][filter]))
        stimtype_series = df[column_name][filter]
        current_sess_y = np.empty(shape=(session_effective_size,),dtype='object')
        for class_ind, current_class in enumerate(classes):
            class_filter = np.array(stimtype_series.isin(current_class))
            current_sess_y[class_filter] = class_labels[class_ind]
        y.extend(current_sess_y)

    # recast everything
    y = np.array(y)
    session = np.array(session)
    modality = np.array(modality)
    beta_numbers = np.array(beta_numbers)

    permutations_path = op.join(permutations_dir,subject+'_permutations.jl')
    joblib.dump(all_permuts,permutations_path,compress=3)

    classif_metadata_path = op.join(classif_metadata_dir,subject+'_classif_metadata.jl')
    joblib.dump([y,session,modality,beta_numbers],classif_metadata_path,compress=3)

