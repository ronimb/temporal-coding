import pickle
import dill

def save_obj(obj, location, save_module=pickle):
    with open(location, 'wb') as file:
        pickle.dump(obj, location, protocol=pickle.HIGHEST_PROTOCOL)