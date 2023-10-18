import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import datetime
import cv2

print(str(datetime.datetime.now()) + ' Initializing')
batch_size = 16
img_height = 640
img_width = 360

# builder = tfds.ImageFolder('resources/datasets/00009-dataset')
train_ds = tf.keras.utils.image_dataset_from_directory(
        pathlib.Path('resources/datasets/00009-dataset'),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
# Загружаем данные об архитектуре сети из файла json
print(str(datetime.datetime.now()) + " Загрузка данных об архитектуре сети")
json_file = open("00009_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("00009_model.h5")
print(str(datetime.datetime.now()) + " Загрузка сети завершена")

# Компилируем загруженную модель
loaded_model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

cap = cv2.VideoCapture('resources/video/00008.MTS')

try:
    while cap.isOpened():
        success, frame = cap.read()
        if success is True:
            img = cv2.resize(frame, (img_width, img_height))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = loaded_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            cv2.putText(frame,
                        "This frame most likely belongs to {} with {:.2f} percent confidence."
                        .format(class_names[np.argmax(score)], 100 * np.max(score)),
                        (30, 50), cv2.FONT_ITALIC, 1, (0, 255, 255), 2, cv2.LINE_AA)

            frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
            cv2.imshow('Stream', frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
finally:
    print()