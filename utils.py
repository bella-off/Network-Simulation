import pickle
from tqdm_thread import tqdm_thread

def dump_data(data, path):
    with tqdm_thread(desc="dumping data {}".format(path)):
        file = open(path, "wb")
        pickle.dump(data, file)
        file.close()

def read_data(path):
    with tqdm_thread(desc="reading data {}".format(path)):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
    return data
