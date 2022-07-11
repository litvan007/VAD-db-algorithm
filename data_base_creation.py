import logging
import os
import pickle
import librosa
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from multiprocessing import Pool
from multiprocessing import Manager

import numpy as np

original_path = '/Users/litvan007/original_2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class data_base_create:
    def __init__(self, original_path='/Users/litvan007/original_2', SNR_db_arr=[1, 3], sr=8000, max_t_signal=6, min_t=0.9, max_t=1.3):
        logger.info('Start creating.')
        self.sr = sr
        self.path = original_path
        self.min_t = min_t*self.sr
        self.max_t = max_t*self.sr
        self.max_t_signal = max_t_signal*self.sr
        self.max_t_input_signal = self.max_t_signal - self.max_t - self.min_t
        self.SNR_db_arr = SNR_db_arr
        self.data = Manager().list()


    def __del__(self):
        logger.info('The process is finished')

    def float32_to_int16(self, x):
        return

    def size_of_noise(self, min_t, max_t):
        return round(np.random.uniform(min_t, max_t, 1)[0])

    def create_noise(self, SNR_db, p, noise_size):
        snr = 10.0 ** (SNR_db / 10.0)
        sample_noise = np.random.normal(0, np.sqrt(p / snr), noise_size)
        return sample_noise

    def sound_create(self, file):
        # print(file)
        arr, sr = librosa.load(f'{self.path}/{file}', sr=self.sr)
        p = np.var(arr)

        if arr.size > self.max_t_input_signal: # > 6.5c
            # print(file)
            return

        num = round(np.random.uniform(0, len(self.SNR_db_arr)-1, 1)[0])

        first_noise = self.create_noise(self.SNR_db_arr[num], p, self.size_of_noise(self.min_t, self.max_t))
        last_noise = self.create_noise(self.SNR_db_arr[num], p, self.max_t_signal - first_noise.size - arr.size)
        noise = self.create_noise(self.SNR_db_arr[num], p, arr.size)

        arr_temp = np.concatenate([first_noise, arr+noise, last_noise])
        marks = np.concatenate([np.zeros(first_noise.size), np.ones(arr.size), np.zeros(last_noise.size)]).astype('int8')

        if arr_temp.size > self.max_t_signal: # > 6c
            arr_temp = arr_temp[:self.max_t_signal]
            marks = marks[:self.max_t_signal]

        # print(first_noise.size, arr.size, last_noise.size, arr_temp.size)
        file_name = f'{file.split(".")[0]}_SNR_{self.SNR_db_arr[num]}'
        temp = {'name': file_name, 'marks': marks}
        self.data.append(temp)
        write(f'./data/{file_name}.wav', self.sr, arr_temp.astype('float32'))

    def creation(self):
        logger.info('Creating...')
        for dirs, folders, files in os.walk(self.path):
            k = len(files)
            with Pool(os.cpu_count()) as p:
                p.map(self.sound_create, files)

        logger.info('Form pickle file')
        with open(f'./data_base.pickle', 'wb') as f:
            pickle.dump(list(self.data), f)
