from sklearn.datasets import fetch_openml
import numpy as np
# Загружаем MNIST
X, Y = fetch_openml("mnist_784",
                    version=1,
                    return_X_y=True,
                    as_frame=False,
                    parser="auto") # type: ignore

# Успокаиваем Pylance — говорим, что это numpy массивы
X: np.ndarray = X  # type: ignore
Y: np.ndarray = Y  # type: ignore

# Преобразуем типы
X = X.astype("float32") / 255.0
y = Y.astype("int64")

# Разделяем
X_training = X[:60000]
X_test = X[60000:]
y_training = y[:60000]
y_test = y[60000:]