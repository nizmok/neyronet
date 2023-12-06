import numpy as np
import pandas as pd
import tensorflow as tf
import keras as keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Генерация примера временного ряда

df = pd.read_csv('E:/Data/SBER/SBER_05122023_06122023.csv')
print(df.shape, df.columns)

#df = df.drop('<TIME>',axis=1)          # 'это если нужно убрать ненужные колонки
#print(df)

# здесь выбор колонок для нейросети и разделение на обучающую и тестовую выборки

split = 0.8
i_split = int(len(df) * split)

cols = ['<CLOSE>']
data_train = df.get(cols).values[:i_split]
data_test = df.get(cols).values[i_split:]
len_train = len(data_train)
len_test = len(data_test)
print(len(df), len_train, len_test)

# Нормализация данных

scaler = MinMaxScaler(feature_range=(-1, 1))    # либо просто MinMaxScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_test_scaled = scaler.fit_transform(data_test)

# Подготовка данных для обучения
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 10                                  # look_back это значений в строке
X, y = create_dataset(data_train_scaled, look_back)   # X [10, n ] это массив основной, y [n] это массив поверочный
X_test, y_test = create_dataset(data_test_scaled, look_back)   # тестовый

#print(X)
print(X.shape)
#print(y)
print(y.shape)


# Решейп для входа в LSTM сеть [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X.shape)

# Создание модели LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))

#model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, epochs=10, batch_size=1, verbose=2)

# Проверка на тестовом наборе
model.evaluate(X_test, y_test, verbose=2)

# Предсказание на тестовом наборе данных
train_predict = model.predict(X_test)

# Инвертирование нормализации для отображения результатов
train_predict = scaler.inverse_transform(train_predict)
y_test = scaler.inverse_transform([y_test])

# Визуализация результатов
plt.plot(data_test, label='Исходный временной ряд')
plt.plot(np.arange(look_back, len(train_predict) + look_back), train_predict, label='Предсказание')
plt.legend()
plt.show()

