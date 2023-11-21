
import tensorflow as tf

# загружаю стандартный набор картинок чисел из библиотеки керас
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормализация данных делю каждый пиксель на максимальное значение  255
x_train, x_test = x_train / 255.0, x_test / 255.0

# создаю модель нейро сети
# 1 слой входной по размеру картинки
# 2 свертка скрытый слой
# 3 прореживание
# 4 выходной - 10 цифр

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# вывод полученой модели
model.summary()
