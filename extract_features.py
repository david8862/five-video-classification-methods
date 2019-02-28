"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm

def save_seq_txt(txt_path, sequence):
    sequence = np.asarray(sequence)
    #print(a.shape)

    #flatten the sequence for save
    flat_sequence = sequence.reshape(-1)

    with open(txt_path, 'w') as f:
        #save sequence length and feature length
        #in 1st two lines
        f.write(str(sequence.shape[0])+'\n')
        f.write(str(sequence.shape[1])+'\n')

        for item in flat_sequence:
            f.write(str(item)+'\n')


# Set defaults.
seq_length = 5
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = os.path.join('data', 'sequences', video[2] + '-' + str(seq_length) + \
        '-features')  # numpy will auto-append .npy

    txt_path = os.path.join('data', 'sequences_txt', video[2] + '-' + str(seq_length) + \
        '-features.txt')

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence.
    np.save(path, sequence)
    # Save the sequence to txt format for tflite inference.
    save_seq_txt(txt_path, sequence)

    pbar.update(1)

pbar.close()
