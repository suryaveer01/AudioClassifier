import argparse
import glob
import os

import tensorflow as tf
import librosa


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array



def parse_audio_file(audio_file: str):
    y, sr = librosa.load(audio_file, sr=44100, mono=True)

    filename = audio_file.split("/")[-1]
    label = int(filename.split("-")[1])

    return y, sr, label, filename

def parse_single_audio_file(y, sr, label, filename):
    data = {
        "sr": _int64_feature(sr),
        "len": _int64_feature(len(y)),
        "y": _bytes_feature(serialize_array(y)),
        "label": _int64_feature(label),
        "filename": _bytes_feature(serialize_array(filename)),
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out



def write_audio_to_tfr(audio_files, filename: str = "audio"):
    filename = filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(
        filename
    )  # create a writer that'll store our data to disk
    count = 0

    for audio_file in audio_files:
        audio_data, sampling_rate, label, filename = parse_audio_file(audio_file)
        # define the dictionary -- the structure -- of a single example
        out = parse_single_audio_file(
            y=audio_data, sr=sampling_rate, label=label, filename=filename
        )
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def main(args):

    # create the output directory
    os.makedirs(args.out_dir, exist_ok=True)

    for fold_number in range(1, 11):
        current_pattern = os.path.join(args.dataset_dir, f"fold{fold_number}/*.wav")
        print(f"Currently parsing fold {fold_number} with pattern {current_pattern}")
        audio_files = glob.glob(current_pattern)
        print(f"Found {len(audio_files)} matching files")
        out_path = os.path.join(args.out_dir, f"fold_{fold_number}")
        write_audio_to_tfr(audio_files=audio_files, filename=out_path)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the folder containing the audio files. Usually, ESC-50-Master/audio/",
    )
    argument_parser.add_argument(
        "--out_dir",
        type=str,
        help="Directory where the TFRecord files are stored",
    )

    arguments = argument_parser.parse_args()
    main(args=arguments)