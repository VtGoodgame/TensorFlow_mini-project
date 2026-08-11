from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

x_mnist, y_mnist = fetch_openml("mnist_784",
                                version=1,
                                return_X_y=True,
                                as_frame=False,
                                parser="auto")

# Разделение на train и test (MNIST: первые 60000 — train, остальные — test)
X_training, X_test, y_training, y_test = train_test_split(
    x_mnist, y_mnist, test_size=10000, random_state=42, stratify=y_mnist
)