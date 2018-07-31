import numpy as np
import pickle
from make_test_samples import gen_with_vesicle_release, convert_multi_samples
import os
import re
import time
from datetime import datetime
# %%
root_folder = '/mnt/disks/data/18_07_samples'
source_folder = os.path.join(root_folder, 'Sources')
save_folder = os.path.join(root_folder, 'mock_identical')
num_samps = 5
spans = [3, 6, 9]
if not (os.path.exists(save_folder)):
    oldmask = os.umask(0)
    os.mkdir(save_folder, 0o777)
    os.umask(oldmask)
# %%
for freq_folder in os.listdir(source_folder):
    freq = int(re.findall(r'(\d+)hz', freq_folder)[0])
    done = []
    for file in os.listdir(os.path.join(source_folder, freq_folder)):
        num_neur = int(re.findall(r'(\d+)Neurons', file)[0])
        if num_neur not in done:
            print(f"Working on {freq}Hz, {num_neur} Neurons")
            done.append(num_neur)
            samples = np.load(os.path.join(source_folder, freq_folder, file))
            for span in spans:
                print(f'span={span}')
                target_folder = os.path.join(save_folder,
                                             f'num_neur={num_neur}_rate={freq}_span={span}')
                if not (os.path.exists(target_folder)):
                    oldmask = os.umask(0)
                    os.mkdir(target_folder, 0o777)
                    os.umask(oldmask)
                for i in range(num_samps):
                    print(f"\tSample {i+1}/{num_samps}")
                    sample = samples[i]
                    shift_sources = [sample['a'], sample['a']]
                    vesrel = gen_with_vesicle_release(rate=freq, num_neur=num_neur,
                                                      span=span, mode=1, num_ves=20, set1_size=100, set2_size=100,
                                                      source_stims=shift_sources)
                    vesrel = convert_multi_samples(vesrel)
                    with open(os.path.join(target_folder, f'set{i}.pickle'), 'wb') as f:
                        pickle.dump(vesrel, f, pickle.HIGHEST_PROTOCOL)
            print('done')


