import tensorflow as tf
import os
from PIL import Image 
import numpy as np

dataset_dir = 'Dataset/original_images/'

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    # Preprocess image as needed (e.g., normalization)
    image = np.array(image) / 255.0  # Normalize to [0,1]
    return image

# Function to load image filenames from directory
def load_image_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other extensions if needed
            filenames.append(os.path.join(directory, filename))
    return filenames

# Load image filenames
image_filenames = load_image_filenames(dataset_dir)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_to_tfrecord(images, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image in images:
            # Convert image to raw bytes
            image_raw = tf.io.encode_jpeg(image)

            # Create a feature
            feature = {
                'image': _bytes_feature(image_raw)
            }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write to TFRecord
            writer.write(example.SerializeToString())

# Example usage
# Iterate through image filenames and preprocess images
train_images = []
for image_path in image_filenames:
    image = load_and_preprocess_image(image_path)
    train_images.append(image)
  # List of training images

# Convert training data to TFRecord
convert_to_tfrecord(train_images, 'train_data.tfrecords')

# Convert testing data to TFRecord
convert_to_tfrecord(train_images, 'test_data.tfrecords')
