import random
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import Model
from keras.models import load_model
import matplotlib.pyplot as plt


def randX():
    x = round(random.uniform(-5, 10), 2)
    return x if x != 0 else randX()  # т.к. в формуле признака 2 присутствует логарифм


def randE():
    return round(random.uniform(0, 0.3), 2)


size = 1000 # размер выборки

feature = [[] for _ in range(6)] # 6 признаков
target = [] # целевое значение

for _ in range(size):
    x = randX()
    e = randE()

    feature[0].append(- x ** 3 + e)
    feature[1].append(np.log(np.abs(x)) + e)
    feature[2].append(np.sin(3 * x) + e)
    feature[3].append(np.exp(x) + e)
    feature[4].append(x + 4 + e)
    feature[5].append(- x + np.sqrt(np.abs(x)) + e)
    target.append(x + e)

data = pd.DataFrame({
    "Признак 1" : feature[0],
    "Признак 2" : feature[1],
    "Признак 3" : feature[2],
    "Признак 4" : feature[3],
    "Признак 5" : feature[4],
    "Признак 6" : feature[5],
    "Цель" : target
})


data.head()
data.to_csv("data_lab5.csv")


data = pd.read_csv("data_lab5.csv", index_col=0)
data.head()

div = round(size*0.8)  # 0.8 - тренировочная, 0.2 - тестовая

train_feature = data.iloc[:div, 0:6]  # срезы выборок
test_feature = data.iloc[div:, 0:6]

train_target = data.iloc[:div, 6:7]  # срезы целевых значений (target)
test_target = data.iloc[div:, 6:7]


# Входной слой
main_input = Input(shape=(6,), dtype="float32", name="main_input")
# Слой кодирования
coder_output = Dense(8, activation="relu", name="coder_output")(main_input)

# Скрытые слои
x = Dense(16, activation="relu")(coder_output)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(4, activation="relu")(x)

# Выходной слой регрессионной модели
decoder_output = Dense(1, activation="sigmoid", name="decoder_output")(x)

# Скрытые слои
x = Dense(16, activation="relu")(coder_output)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)

# Выходной слой регрессионной модели
regression_output = Dense(1, name="regression_output")(x)

model = Model(inputs=[main_input], outputs=[regression_output, decoder_output])
model.compile(
    optimizer="rmsprop",
    loss="mean_squared_logarithmic_error",
    metrics=['accuracy'],
    loss_weights=[1., 1.]
)

H = model.fit(
    train_feature,
    train_target,
    epochs=100,
    batch_size=16,
    validation_data=(
        test_feature,
        test_target
    )
)

loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', color="red", label='Training loss')
plt.plot(epochs, val_loss, 'bo', color="blue", label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('lab5.h5') # Выгрузка модели
del model # Удаление модели из оперативной памяти
model = load_model('lab5.h5') # Загрузка модели

pred = model.predict(test_feature)

out = pd.DataFrame()
out["Полученное"] = [item for sublist in pred[0].tolist() for item in sublist]
out["Исходное"] = test_target.Цель.tolist()
out.head()


out.to_csv("data_lab5.csv")

