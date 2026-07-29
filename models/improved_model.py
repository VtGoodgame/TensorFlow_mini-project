import datetime
from tensorflow import keras
from tensorflow.keras import layers

from data.iris import X_test, X_train, y_train, y_test
from handlers.save_handler import save_results, save_model

improved_model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),   # больше нейронов
    layers.Dropout(0.3),                  # отключаем 30% нейронов
    layers.Dense(8, activation="relu"),   # дополнительный слой
    layers.Dropout(0.3),                  # ещё отключаем 30%
    layers.Dense(3, activation="softmax")
])

improved_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
improved_history = improved_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=0
)

# Оценка на тесте
improved_test_loss, improved_test_acc = improved_model.evaluate(
    X_test, y_test, verbose=1
)
improved_train_loss, improved_train_acc = improved_model.evaluate(
    X_train, y_train, verbose=1
)

# Собираем метрики
results = {
    "model_name": "improved_model",
    "test_accuracy": float(improved_test_acc),
    "test_loss": float(improved_test_loss),
    "train_accuracy": float(improved_train_acc),
    "train_loss": float(improved_train_loss),
    "val_accuracy": float(improved_history.history["val_accuracy"][-1]),
    "val_loss": float(improved_history.history["val_loss"][-1]),
    "total_params": improved_model.count_params(),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

#  Сохраняем результаты и модель
save_results(results = results, model_name = "improved_model")
save_model(model = improved_model, model_name = "improved_model")