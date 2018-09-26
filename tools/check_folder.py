import os
def check_folder(folder_location):
    # This is a simple function for checking if a folder exists, and creating it if it does not
    if not (os.path.exists(folder_location)):
        oldmask = os.umask(0)
        os.mkdir(folder_location, 0o777)
        os.umask(oldmask)
