import os
import json
import math
import librosa

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


class pre_process():
    def __init__(self, num_mfcc=13, num_fft=2048, hop_length=512, num_segments=5):
        # * Save input variables
        self.num_mfcc = num_mfcc
        self.num_fft = num_fft
        self.num_segments = num_segments
        self.hop_length = hop_length

        # * Set default varibales
        self.samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        self.num_mfcc_vectors_per_segment = math.ceil(self.samples_per_segment / hop_length)

        # * Structure to store every data
        self.data = {"mapping": [], "labels": [], "mfcc": []}

    def process_data(self, data_path, verbose=1):
        # * Check every directory inside data_path
        for label, (dir_path, _, file_list) in enumerate(os.walk(data_path)):
            if label != 0:
                genre = dir_path.split('/')[-1]
                self.data['mapping'].append(genre)
                if verbose:
                    print("Processing:", genre)

                # * Check every file inside genre subdir
                for file_name in file_list:
                    file_path = os.path.join(dir_path, file_name)

                    # * Load audio file
                    signal, sample_rate = librosa.load(
                        file_path, sr=SAMPLE_RATE)
                    for segment in range(self.num_segments):
                        # * Start/Finish sample for current segment
                        start_sample = self.samples_per_segment * segment
                        finish_sample = start_sample + self.samples_per_segment

                        # * Extract mfcc
                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                    sr=sample_rate,
                                                    n_mfcc=self.num_mfcc,
                                                    n_fft=self.num_fft,
                                                    hop_length=self.hop_length)
                        mfcc = mfcc.T

                        # * Store mfcc feature with expected number of vectors : for input of machine
                        if len(mfcc) == self.num_mfcc_vectors_per_segment:
                            self.data['labels'].append(label - 1)
                            self.data["mfcc"].append(mfcc.tolist())
                            if verbose >= 2:
                                print("{}, segment:{}".format(
                                    file_path, segment + 1))

    def save(self, json_path):
        with open(json_path, "w") as file:
            json.dump(self.data, file, indent=4)
