from tensorflow import keras
from tensorflow.keras import layers
from data.iris import X_test, X_train, y_train, y_test

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
    X_test, y_test, verbose=0
)
improved_train_loss, improved_train_acc = improved_model.evaluate(
    X_train, y_train, verbose=0
)