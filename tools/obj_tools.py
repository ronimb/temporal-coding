import pickle

def save_obj(obj, location):
    with open(location, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(obj, location):
    with open(location, 'rb') as file:
        return pickle.load(file)