import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import datetime

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16
img_height = 640
img_width = 360


def model_upload():
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    data_dir = pathlib.Path('resources/datasets/00009-dataset')

    print(str(datetime.datetime.now()) + ' Start reading the dataset')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    class_names = train_ds.class_names
    print(class_names)

    # Кэширование изменяет pipeline
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 3
    epochs = 5

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 360, 3)),
        tf.keras.layers.MaxPooling2D((8, 8)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((4, 4)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_ds
    )

    # model.summary()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Генерируем описание модели в формате json
    model_json = model.to_json()
    # Записываем модель в файл
    model.save_weights("00009_model.h5")
    json_file = open("00009_model.json", "w")
    json_file.write(model_json)
    json_file.close()


if __name__ == '__main__':
    print(str(datetime.datetime.now()) + ' Initializing')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model_upload()


