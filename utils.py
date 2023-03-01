
import numpy as np

from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
from typing import Tuple

data_dir = Path("./samples/")


# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split("/")[-1][:-4].split("samples")[1].replace("\\",'') for img in images]
characters = sorted(list(set(char for label in labels for char in label)))

# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

def split_data(images:list, labels:list, train_size: float=0.9, shuffle:bool =True )->Tuple[list,list,list,list]:
    """
    :param images: list of images
    :param labels: list of labels
    :param train_size: ratio of spliting - it mean we split our data to 90% for training and 10% for validation  
    :param shuffle: shuffle data 
    :return: x_train, x_valid, y_train, y_valid
    """
    
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid



# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))




def encode_single_sample(img_path:str, label:str)->dict:
    """
    :param img_path: directory of image
    :param labels: labele of image
    :return: Dictionary of image and label
    """
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [50, 200])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


def getData(batch_size:int):
    """
    :param batch_size: number of batch size
    :return train_dataset and validation_dataset: a new instance of PrefetchDataset
    """
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return train_dataset,validation_dataset

def getCharacters()->list:
    """
    :return: list of Characters
    """
    return characters