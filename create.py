from data_base_creation import data_base_create
import pickle

if __name__ == '__main__':
    process = data_base_create()
    process.creation()
    data_new = None
    with open('data_base.pickle', 'rb') as fh:
        data_new = pickle.load(fh)
    sound1 = data_new[100]
    sound2 = data_new[200]
    print(sound1['marks'].size, sound2['marks'].size)


