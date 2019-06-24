import os
import os.path as op

subjects_list = ['homofaber_00', 'homofaber_01', 'homofaber_02', 'homofaber_03', 'homofaber_04',
                 'homofaber_05', 'homofaber_06', 'homofaber_07', 'homofaber_08', 'homofaber_09',
                 'homofaber_11', 'homofaber_12', 'homofaber_13', 'homofaber_14', 'homofaber_16']

n_splits = len(subjects_list)

modality_list = ['A','V']

searchlight_radius = 9.

for split_ind, subject in enumerate(subjects_list):
    for train_modality in modality_list:
        for test_modality in modality_list:
            single_split_path1 = op.join(single_split_res_dir,
                                        'intersubj_balancedacc_rad{:05.2f}mm_train{}test{}_split{:1d}of{:1d}.nii.gz'.format(
                                        searchlight_radius, train_modality, test_modality, split_ind + 1, n_splits))
            single_split_path2 = op.join(single_split_res_dir,
                                        'intersubj_balancedacc_rad{:05.2f}mm_train{}test{}_split{:02d}of{:02d}.nii.gz'.format(
                                        searchlight_radius, train_modality, test_modality, split_ind + 1, n_splits))
            os.rename(single_split_path1,single_split_path2)