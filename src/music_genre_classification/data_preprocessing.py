import json
import os
import math
import librosa

DATASET_PATH = "genres_dataset"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(DATASET_PATH,JSON_PATH, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    data = {
        "genres": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

        if dirpath is not DATASET_PATH:

            semantic_label = os.path.basename(dirpath)
            data["genres"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            for filename in filenames:

                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):

                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print(f"{file_path}, segment:{d+1}")

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
