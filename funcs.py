import tensorflow as tf
import pathlib
import os
import numpy as np
import shutil


def find_max_len(dataset):
    # Find file with maximum audio length
    audio_max_len = 0
    audio_max = 0
    dataset = dataset.enumerate()

    for i, (audio, _) in dataset.as_numpy_iterator():
        audio_len = len(audio)
        if audio_len > audio_max_len:
            audio_max_len = audio_len
            audio_max = audio

    return audio_max_len, audio_max


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    # Get label of file from our particular data
    parts = tf.strings.split(file_path, '/')
    label = tf.strings.split(parts, '_')
    return label[-1][0]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 48000 samples
    zero_padding = tf.zeros([48000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=130)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.strings.to_number(label, out_type=tf.int32)
    return spectrogram, label_id


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns) and plot.

    log_spec = np.log(spectrogram.T)
    width = 48000 / log_spec.shape[1]
    X = np.arange(48000, step=width)
    Y = np.array(range(log_spec.shape[0]))
    ax.pcolormesh(X, Y, log_spec)


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label,
                             num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)

    return output_ds
