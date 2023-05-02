import tensorflow as tf


def parse_tfr_audio_element(element):

    data = {
        "sr": tf.io.FixedLenFeature([], tf.int64),
        "len": tf.io.FixedLenFeature([], tf.int64),
        "y": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    sr = content["sr"]
    len = content["len"]
    y = content["y"]
    label = content["label"]
    filename = content["filename"]

    # get our 'feature'-- our audio data -- and reshape it appropriately
    feature = tf.io.parse_tensor(y, out_type=tf.float32)
    feature = tf.reshape(feature, shape=[len])
    filename = tf.io.parse_tensor(filename, out_type=tf.string)
    return feature, label, sr, filename


def get_audio_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_audio_element)

    return dataset


ds = get_audio_dataset("path/to/tfr_file.tfrecords")  # can also be multiple files
padded_ds = ds.padded_batch(
    batch_size=16,
    padded_shapes=(
        [
            1764000,
        ],
        [],
        [],
        [],
    ),
    padding_values=(0.0, tf.cast(0, dtype=tf.int64), tf.cast(0, tf.int64), "0"),
)

for sample in ds.take(5):
    print(sample)