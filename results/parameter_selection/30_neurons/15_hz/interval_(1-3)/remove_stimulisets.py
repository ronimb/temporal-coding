import pickle
import os
from tools import check_folder, load_obj, save_obj
# %%
wd = '/home/ronimber/PycharmProjects/temporal-coding/Results/parameter_selection/30_neurons/15_hz/interval_(1-3)'
files = os.listdir(wd)

def fix_experiment(file_loc):
    full_path = os.path.join(wd, file_loc)
    if 'params.experiment' in file_loc:
        exp_params = load_obj(full_path)
        if 'stimuli_sets' in exp_params:
            exp_params.pop('stimuli_sets')
            save_obj(exp_params, full_path)

# %%
for file in files:
    fix_experiment(file)