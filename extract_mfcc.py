import json
import os
import math
import librosa
import sklearn
import numpy as np

DATASET_PATH = "C:\\Users\\dcsri\\PycharmProjects\\IndianMusicGithub\\dataset"
#DATASET_PATH = "C:\\Users\\dcsri\\PycharmProjects\\IndianMusicGithub\\dataset"
JSON_PATH = "data_IndianMusicAll.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "song":[],
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments) #66150 for 5 seg and 15 sec songs
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length) #1.2
    print (num_mfcc_vectors_per_segment)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)

                #for d in range(num_segments):
                # extract mfcc
                mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc_delta = (librosa.feature.delta(mfcc)).T
                mfcc_delta2 = (librosa.feature.delta(mfcc, order=2)).T
                mfcc = mfcc.T
                print (mfcc.shape, mfcc_delta.shape, mfcc_delta2.shape)
                features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=1)


                # store only mfcc feature with expected number of vectors
                mfcc_mean = np.array(features.mean(axis=0)).reshape(1,-1)
                mfcc_std  =  np.array(features.std(axis=0)).reshape(1,-1)
                mfcc_max = np.array(features.max(axis=0)).reshape(1,-1)
                mfcc_min = np.array(features.min(axis=0)).reshape(1,-1)
                features = np.concatenate((mfcc_mean, mfcc_std, mfcc_max, mfcc_min), axis=1)
                print(features.shape)
                data["mfcc"].append(features.tolist())
                data["labels"].append(i - 1) # each i corresponds to a genre folder
                print (f,f.split(".wav")[0])
                data["song"].append(f.split(".wav")[0])
                print("{}".format(file_path))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)