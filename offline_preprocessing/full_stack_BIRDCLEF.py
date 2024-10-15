###############################################################################
# IMPORTS AND PARAMATERS
###############################################################################
import os

# Grabs all of the relevent main fucntions from associated scripts
from to_var_spec import main as to_var_spec_main
from to_np_and_norm import main as np_norm_main
from folder_scripts.pruning_BirdClef import main as pruning_main

MAIN_DIR = '/data/BirdClef/audio/audio'
# Path wated for final spectrogram data
FINAL_SPEC_PATH = MAIN_DIR + '/features'

NORM = True
SAMPLE_LENGTH = 5
MEL_SPEC_PARAMS = {'sr': 16000,
                'n_mels':128,
                'n_fft':1024,
                'hop_length':512,
                'power':2.0}

if __name__ == '__main__':
    # Sorts the mixed up audio files into classes
    sorted_path = MAIN_DIR
    # Generates a new path for the sorted classes in npy format
    sorted_npy_path = os.path.join(MAIN_DIR, sorted_path + '_npy')
    #np_norm_main(old_dir=sorted_path, new_dir=sorted_npy_path, sr=MEL_SPEC_PARAMS['sr'], norm=NORM)

    pruning_main(main_dir=sorted_npy_path,
    time_thresh_s=180, # Maximum time in secinds
    class_thresh=50, # Min num samples per class
    SR=MEL_SPEC_PARAMS['sr'], #Sample rate
    remove=True) #Should we remove the samples yet

    # Carries out the spectrogram dataste creation
    to_var_spec_main(old_dir=sorted_npy_path,
                     new_dir=FINAL_SPEC_PATH,
                     sample_length=SAMPLE_LENGTH,
                     spec_params=MEL_SPEC_PARAMS)
