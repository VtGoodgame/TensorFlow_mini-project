from tensorflow import keras
from tensorflow.keras import layers
from data.iris import X_test, X_train, y_train, y_test

# 1. Создаём модель — полносвязная сеть
baseline_model = keras.Sequential([
    layers.Input(shape=(4,)),           # Вход: 4 признака
    layers.Dense(8, activation="relu"), # Скрытый слой: 8 нейронов, relu - что то вроде активация при отклонениях
    layers.Dense(3, activation="softmax") # Выход: 3 нейрона (3 класса), Softmax работает с использованием экспоненты
])

# 2. Компилируем
baseline_model.compile(
    optimizer="adam", # Оптимизатор — это метод, которым сеть учится на своих ошибках.
    loss="sparse_categorical_crossentropy", # Функция потерь — это измеритель ошибки. 
                                # Categorical crossentropy — для задач, где нужно выбрать один класс из нескольких (у нас 3 ириса).
                                # Sparse — значит, что метки у нас в виде одного числа
    metrics=["accuracy"] # Accuracy = точность, доля правильных ответов. Отображение прогресса в консоли
)

# 3. Смотрим, что получилось
baseline_model.summary()

# 4. Обучаем
history = baseline_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1 # убираем вывод
)

# Оценка на тесте
baseline_test_loss, baseline_test_acc = baseline_model.evaluate(
    X_test, y_test, verbose=0
)
baseline_train_loss, baseline_train_acc = baseline_model.evaluate(
    X_train, y_train, verbose=0
)