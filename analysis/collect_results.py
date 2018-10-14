import os
import pandas as pd
# %%

def collect_results(results_path, pattern='results.csv', with_subfolders=True,
                    save_path='',  # Default is to only return the dataframe without saving
                    save_name='combined_results',
                    save_formats=['csv', 'df']): # If saving, you can get both the pandas dataframe and the csv saved, or either one
    # Read data from all file files matching pattern to pandas csv
    all_data = []
    matching_files = []  # For debugging
    for results_path, subdirs, files in os.walk(results_path):
        for file in files:
            # Check if the file contains the specified pattern
            if pattern in file:
                file_path = os.path.join(results_path, file)
                matching_files.append(file)  # For debugging
                all_data.append(pd.read_csv(file_path, index_col=0))

    # Merge to a single data frame
    combined_data = pd.concat(all_data, ignore_index=True, sort=False)

    if save_path:
        file_name = os.path.join(save_path, save_name)
        if 'csv' in save_formats:
            combined_data.to_csv(f'{file_name}.csv')
        if 'df' in save_formats:
            combined_data.to_pickle(f'{file_name}.pkl')
    return combined_data
# %%
if __name__ == '__main__':
    results_path = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/Results/Experiments/'
    save_path = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/Results'
    save_name = 'all_experiment_results'

    collect_results(results_path=results_path, save_name=save_name, save_path=save_path)
