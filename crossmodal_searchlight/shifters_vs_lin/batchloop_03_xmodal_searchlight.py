import subprocess

code_root_dir = '/envau/userspace/takerkart/python/homofaber_mvpa/homofaber_ispa/crossmodal_searchlight/shifters_vs_lin'

subjects_list = ['homofaber_00', 'homofaber_01', 'homofaber_02', 'homofaber_03', 'homofaber_04',
                 'homofaber_05', 'homofaber_06', 'homofaber_07', 'homofaber_08', 'homofaber_09',
                 'homofaber_11', 'homofaber_12', 'homofaber_13', 'homofaber_14', 'homofaber_16']

modality_list = ['A','V']

searchlight_radius = 9.

for split_ind, subject in enumerate(subjects_list):
    for train_modality in modality_list:
        for test_modality in modality_list:
            cmd = "frioul_batch -c 12 '/hpc/crise/anaconda3/bin/python {}/03_homofaber_intersubject_searchlight_crossmodalities_decoding.py {:02d} {} {} {:1.2f}'".format(code_root_dir, split_ind, train_modality, test_modality, searchlight_radius)
            print(cmd)
            subprocess.run(cmd,shell=True)
