import datetime
from tensorflow import keras
from tensorflow.keras import layers

from data.iris import X_test, X_train, y_train, y_test
from handlers.save_handler import save_results, save_model

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
    verbose=0 # убираем вывод
)

# Оценка на тесте
baseline_test_loss, baseline_test_acc = baseline_model.evaluate(
    X_test, y_test, verbose=1
)
baseline_train_loss, baseline_train_acc = baseline_model.evaluate(
    X_train, y_train, verbose=1
)

# Собираем метрики
results = {
    "model_name": "Baseline",
    "test_accuracy": float(baseline_test_acc),
    "test_loss": float(baseline_test_loss),
    "train_accuracy": float(baseline_train_acc),
    "train_loss": float(baseline_train_loss),
    "val_accuracy": float(history.history["val_accuracy"][-1]),
    "val_loss": float(history.history["val_loss"][-1]),
    "total_params": baseline_model.count_params(),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Сохраняем результаты и модель
save_results(results = results, model_name = "baseline_model")
save_model(model = baseline_model, model_name = "baseline_model")