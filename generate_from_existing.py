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
save_folder = os.path.join(root_folder, 'vesrel')
if not (os.path.exists(save_folder)):
    oldmask = os.umask(0)
    os.mkdir(save_folder, 0o777)
    os.umask(oldmask)
spans = [3, 6, 9]
# %%
for freq_folder in os.listdir(source_folder):
    freq = int(re.findall(r'(\d+)hz', freq_folder)[0])
    for file in os.listdir(os.path.join(source_folder, freq_folder)):
        num_neur = int(re.findall(r'(\d+)Neurons', file)[0])
        distance = float(re.findall(r'distance=(\d\.\d+)', file)[0])
        print(f"Working on {freq}Hz, {num_neur} Neurons at {distance} distance, started at {datetime.now().strftime('%d/%m %H:%M:%S')}")
        samples = np.load(os.path.join(source_folder, freq_folder, file))
        for span in spans:
            target_folder = os.path.join(save_folder,
                                         f'num_neur={num_neur}_rate={freq}_distance={distance}_span={span}')
            if not (os.path.exists(target_folder)):
                oldmask = os.umask(0)
                os.mkdir(target_folder, 0o777)
                os.umask(oldmask)
            samp_times = np.zeros(samples.shape[0])
            for i, sample in enumerate(samples):
                samp_time = time.time()
                print(f'\t Span={span}, sample#{i+1}', end='', flush=True)
                shift_sources = [sample['a'], sample['b']]
                vesrel = gen_with_vesicle_release(rate=freq, num_neur=num_neur,
                                                  span=span, mode=1, num_ves=20, set1_size=100, set2_size=100,
                                                  source_stims=shift_sources)
                vesrel = convert_multi_samples(vesrel)
                with open(os.path.join(target_folder, f'set{i}.pickle'), 'wb') as f:
                    pickle.dump(vesrel, f, pickle.HIGHEST_PROTOCOL)
                samp_times[i] = time.time() - samp_time
                print(f" sample took {time.strftime('%M:%S', time.gmtime(samp_times[i]))}")
            print(f"\tMean generation time for span={span} is {time.strftime('%M:%s', time.gmtime(samp_times.mean()))}")
