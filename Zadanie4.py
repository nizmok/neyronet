
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Шаг 1: Загрузка данных
# 1.1. Загрузка данных CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 1.2. Подготовка данных
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Шаг 2: Создание сверточной нейронной сети
# 2.1. Импорт библиотек
model = models.Sequential()

# 2.2. Создание сверточной нейронной сети
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Шаг 3: Обучение сверточной нейронной сети
# 3.1. Настройка параметров обучения
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3.2. Обучение модели
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Шаг 4: Оценка производительности модели
# 4.1. Оценка на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Точность на тестовых данных:", test_acc)

'''
Epoch 1/10
625/625 [==============================] - 20s 30ms/step - loss: 1.6237 - accuracy: 0.4031 - val_loss: 1.4135 - val_accuracy: 0.4866
Epoch 2/10
625/625 [==============================] - 19s 30ms/step - loss: 1.2642 - accuracy: 0.5464 - val_loss: 1.2058 - val_accuracy: 0.5813
Epoch 3/10
625/625 [==============================] - 18s 29ms/step - loss: 1.1181 - accuracy: 0.6048 - val_loss: 1.0859 - val_accuracy: 0.6189
Epoch 4/10
625/625 [==============================] - 19s 30ms/step - loss: 1.0261 - accuracy: 0.6385 - val_loss: 1.0175 - val_accuracy: 0.6444
Epoch 5/10
625/625 [==============================] - 18s 29ms/step - loss: 0.9467 - accuracy: 0.6693 - val_loss: 0.9550 - val_accuracy: 0.6654
Epoch 6/10
625/625 [==============================] - 18s 29ms/step - loss: 0.8779 - accuracy: 0.6926 - val_loss: 0.9350 - val_accuracy: 0.6793
Epoch 7/10
625/625 [==============================] - 18s 29ms/step - loss: 0.8339 - accuracy: 0.7067 - val_loss: 0.9314 - val_accuracy: 0.6738
Epoch 8/10
625/625 [==============================] - 18s 29ms/step - loss: 0.7789 - accuracy: 0.7284 - val_loss: 0.8982 - val_accuracy: 0.6901
Epoch 9/10
625/625 [==============================] - 18s 30ms/step - loss: 0.7428 - accuracy: 0.7398 - val_loss: 0.8877 - val_accuracy: 0.6944
Epoch 10/10
625/625 [==============================] - 19s 30ms/step - loss: 0.7060 - accuracy: 0.7527 - val_loss: 0.8822 - val_accuracy: 0.7005
313/313 [==============================] - 2s 5ms/step - loss: 0.9031 - accuracy: 0.6966
Точность на тестовых данных: 0.6966000199317932

'''

