from keras.models import Sequential
from keras.layers import Dense

# Создание последовательной модели
model = Sequential()

# Добавление слоя с одним входом и одним выходом (один нейрон) в скрытом слое
model.add(Dense(units=1, input_dim=1))

# Компиляция модели
model.compile(optimizer='sgd', loss='mse') # оптимизатор стохастического градиентного спуска (SGD) и функция потерь среднеквадратичной ошибки (MSE)

# Генерация входных данных и меток
import numpy as np
X = np.array([1, 2, 3, 4, 5]) # входные данные
y = np.array([2, 4, 6, 8, 10]) # метки (ожидаемые выходы)

# Обучение модели
model.fit(X, y, epochs=1000) # количество эпох обучения = 1000

# Предсказание выхода на новых данных
X_test = np.array([6, 7, 8, 9, 10]) # новые данные для предсказания
y_pred = model.predict(X_test) # предсказанные выходы

print(y_pred) # вывод предсказанных выходов
