import pickle


def save_pickle_object(obj, obj_name):
    try:
        with open(obj_name, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise Exception(f"Unable to save pickle object {e}")
    

def load_pickle_object(obj_path):
    try:
        with open(obj_path, 'rb') as f:
            obj = pickle.load(f)

        return obj
    except Exception as e:
        raise Exception(f"Unable to load pickle object {e}")