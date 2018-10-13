import pickle
from os import umask, makedirs
from os.path import exists, dirname, join


# %%
def make_folder(base_folder, *subfolders):
    path = join(base_folder, *subfolders)
    check_folder(path)
    return path

def check_folder(folder_location):
    # This is a simple function for checking if a folder exists, and creating it if it does not
    if folder_location:  # Check if empty string was passed
        if not (exists(folder_location)):
            print(f'creating folder at {folder_location}')
            oldmask = umask(0)
            makedirs(folder_location, 0o777)
            umask(oldmask)
    else:
        pass


def save_obj(obj, location):
    check_folder(dirname(location))
    with open(location, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(location):
    check_folder(dirname(location))
    with open(location, 'rb') as file:
        return pickle.load(file)
