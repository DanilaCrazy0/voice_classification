import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import os
import torch


def get_df(folder_path, mode):
    def get_answers(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            answers = {}
            for line in lines:
                filename, out = line.strip().split('\t')
                answers[filename] = int(out)
        return pd.DataFrame.from_dict(answers, orient='index')


    def cutting_voice(duration, y, sr):
        times = librosa.frames_to_time(range(y.shape[0]), sr=sr)
        print

        duration_n = int(duration / (times[1] - times[0])) + 1
        indx = find_max_index_sum(np.abs(y), duration_n)
        print(y[indx:indx+duration_n])

        return y[indx:indx+duration_n]


    def find_max_index_sum(arr, window_size):
        windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)
        window_sums = np.sum(windows, axis=1)
        max_sum_index = np.argmax(window_sums)
        return max_sum_index


    def get_train(folder_path, file_path):
        data = []
        if mode == 'train':
            answers = get_answers(file_path)
        for audio_path in glob.glob(folder_path):
            y, sr = librosa.load(audio_path, sr=None)
            n_mfcc = 16
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            delta_mfccs = librosa.feature.delta(mfccs)  # Дельта коэффициенты
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)  # Дельта-дельта коэффициенты

            # Объединим все коэффициенты
            mfccs_full = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

            # # Визуализация MFCC
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(mfccs, x_axis='time', sr=sr)
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('MFCC')
            # plt.xlabel('Time')
            # plt.ylabel('MFCC Coefficients')
            # plt.show()

            average = np.mean(mfccs_full, axis=1)
            std = np.std(mfccs_full, axis=1)
            maximum = np.max(mfccs_full, axis=1)
            minimum = np.min(mfccs_full, axis=1)
            median = np.median(mfccs_full, axis=1)

            coeffs = np.array([average, std, maximum, minimum, median]).flatten()

            filename = os.path.basename(audio_path)
            filename = os.path.splitext(filename)[0]

            if mode == 'train':
                data.append([filename, *coeffs.tolist(), answers.loc[filename, 0]])
            else:
                data.append([filename, *coeffs.tolist()])
        return data


    file_path = 'train/targets.tsv'
    folder_path = folder_path
    duration = 1.2

    data = get_train(folder_path, file_path)
    df = pd.DataFrame(data)
    df.set_index(0, inplace=True)
    if mode == 'train':
        df.columns = [f"feature_{i}" for i in range(1, len(df.columns))] + ["target"]
    else:
        df.columns = [f"feature_{i}" for i in range(1, len(df.columns)+1)]
    return df

