# полносвязная модель с использованием Pytorch
# Полносвязная нейронная сеть (Fully Connected Neural Network, FCNN) — 
# это базовая искусственная нейросеть, в которой каждый нейрон текущего слоя
# соединен со всеми нейронами следующего слоя. 
# Сигнал в ней передается только вперед, от входа к выходу,
# 
# через линейные преобразования и функции активации.
import torch
import torch as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml

from data.mnist import X_training, X_test, y_training, y_test

# Преобразование в тензоры
X_train_t = torch.tensor(X_training)
y_train_t = torch.tensor(y_training)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
# DataLoader нужен для загрузки датасета мини пакетами, что более эффективно. 
#  Так же веса будут обновляться после каждого пакета 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Полносвязная модель
class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # Linear - создает матрицу нейронов определяя входными параметрами количество нейронов и колчество выходов
            nn.Linear(784, 256),
            # ReLU - функция активации нейрна, если выход положительный -> return 
            # если отрицательный или 0 -> return 0 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(x)

# 3. Обучение
# torch.cuda.is_available() — проверяет,
# есть ли в системе видеокарта NVIDIA с драйверами CUDA.
# True (GPU доступна) → torch.device('cuda') — будет использоваться видеокарта.
# False → torch.device('cpu') — вычисления пойдут на процессоре.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullyConnected().to(device)

# CrossEntropy специально создана для задач классификации.
criterion = nn.CrossEntropyLoss()

# optim - алгоритм, который обновляет веса модели, чтобы уменьшить loss.
# передает все робучаемые параметры модели И 
# learning rate - скорость обучения, на сколько сильно менять веса
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train() # переводит модель в режим обучения.
    running_loss = 0.0 # переменная для накопления ошибки за всю эпоху.
    # train_loader выдаёт батчи — порции данных. 
    for batch_X, batch_y in train_loader:
        # Перемещает текущий батч на то же устройство, где находится модель.
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # по умолчанию накапливаются градиенты от батча к батчу
        # перед каждым backward() градиенты обнуляют
        optimizer.zero_grad()
        outputs = model(batch_X) # Прямой проход
        loss = criterion(outputs, batch_y) # Вычисление ошибки
        loss.backward() # Обратный проход
        # берёт градиенты из .grad и обновляет веса модели по формуле Adam
        optimizer.step()
        # среднее количество ошибок
        running_loss += loss.item() * batch_X.size(0)

    # Средняя ошибка за эпоху
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# переводим модель в режим оценки (выключаем dropout и batchnorm)
model.eval()
correct = 0 # сколько картинок угадано правильно
total = 0 # сколько всего картинок обработано
with torch.no_grad(): #  отключает вычисление градиентов на всём
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X) # прямой проход по батчу
        _, predicted = torch.max(outputs, 1) # сохранение идекса класса с наибольшей вероятностью   
        total += batch_y.size(0) # прибавляем количество картинок в батче к общему количеству
        correct += (predicted == batch_y).sum().item() # подсчет правильных предсказаний в батче и прибавление к общему количеству правильных предсказаний

accuracy = correct / total # сколько правильных предсказаний от общего количества картинок
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")