import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import datetime

batch_size = 16
img_height = 224
img_width = 224


def model_experiment():
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    data_dir = pathlib.Path('resources/datasets/PillowDataset')

    print(str(datetime.datetime.now()) + ': Start reading the dataset')
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

    class_names = train_ds.class_names
    print(class_names)

    num_classes = 4
    epochs = 20

    inputs = tf.keras.Input(shape=(224, 224, 3), name="img")
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    block_1_output = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(y)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(block_1_output)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    block_2_output = tf.keras.layers.add([y, block_1_output])
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(block_2_output)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    block_3_output = tf.keras.layers.add([y, block_2_output])
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(block_3_output)
    y = tf.keras.layers.GlobalAvgPool2D()(x)
    z = tf.keras.layers.Dense(256, activation="relu")(y)
    u = tf.keras.layers.Dropout(0.5)(z)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(u)

    model = tf.keras.Model(inputs, outputs, name="resnet_model")
    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print(str(datetime.datetime.now()) + ': Start training')
    history = model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_ds
    )
    print(str(datetime.datetime.now()) + ': End training')

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
    model.save_weights("model.h5")
    json_file = open("model.json", "w")
    json_file.write(model_json)
    json_file.close()


if __name__ == '__main__':
    print(str(datetime.datetime.now()) + ': Initializing')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model_experiment()
