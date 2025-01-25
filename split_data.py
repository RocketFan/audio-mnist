import os
import shutil
import random

random.seed(42)

def split_data(source_dir, target_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    if abs((train_ratio + val_ratio + test_ratio) - 1) > 1e6:
        raise ValueError("Ratios must sum to 1")

    train_dir = target_dir + "/train"
    val_dir = target_dir + "/val"
    test_dir = target_dir + "/test"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    speakers = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    random.shuffle(speakers)

    train_count = int(len(speakers) * train_ratio)
    val_count = int(len(speakers) * val_ratio)

    train_speakers = speakers[:train_count]
    val_speakers = speakers[train_count:train_count + val_count]
    test_speakers = speakers[train_count + val_count:]

    for speaker in train_speakers:
        shutil.copytree(os.path.join(source_dir, speaker), os.path.join(train_dir, speaker))

    for speaker in val_speakers:
        shutil.copytree(os.path.join(source_dir, speaker), os.path.join(val_dir, speaker))

    for speaker in test_speakers:
        shutil.copytree(os.path.join(source_dir, speaker), os.path.join(test_dir, speaker))

if __name__ == "__main__":
    source_directory = "data"
    target_directory = "data_split"

    split_data(source_directory, target_directory)