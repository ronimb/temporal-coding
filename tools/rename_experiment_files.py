import string
import os
import re
# %%

files_path = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/Results/'
for maindir, subdirs, files in os.walk(files_path):
    for file in files:
        old_name = file
        if 'interval' in old_name:
            file_type = '_'.join(old_name.split('_')[3:])
            interval, relduration, relprob = [x.split('=')[1] for x in old_name.split('_')[0:3]]
            newname = f'interval_{interval[1]}-{interval[4]}_relduration_{relduration}_prob_{relprob}_{file_type}'
            print(old_name, newname)
