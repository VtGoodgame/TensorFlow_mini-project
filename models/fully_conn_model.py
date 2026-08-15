# полносвязная модель с использованием Pytorch
# Полносвязная нейронная сеть (Fully Connected Neural Network, FCNN) — 
# это базовая искусственная нейросеть, в которой каждый нейрон текущего слоя
# соединен со всеми нейронами следующего слоя. 
# Сигнал в ней передается только вперед, от входа к выходу,
# 
# через линейные преобразования и функции активации.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data.mnist import X_training, X_test, y_training, y_test
from handlers.save_handler import save_results
from handlers.training_result import TrainingResult

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
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Функция оценки: считает средний loss и accuracy на переданном loader
def evaluate_model(model, loader, criterion, device):
    """Вычисляет loss и accuracy на переданном loader."""
    model.eval() # переводим модель в режим оценки (выключаем dropout и batchnorm)
    total_loss = 0.0
    correct = 0 # сколько картинок угадано правильно
    total = 0 # сколько всего картинок обработано
    with torch.no_grad(): #  отключает вычисление градиентов на всём
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X) # прямой проход по батчу
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1) # сохранение индекса класса с наибольшей вероятностью
            total += batch_y.size(0) # прибавляем количество картинок в батче к общему количеству
            correct += (predicted == batch_y).sum().item() # подсчет правильных предсказаний в батче и прибавление к общему количеству правильных предсказаний

    avg_loss = total_loss / total
    accuracy = correct / total # сколько правильных предсказаний от общего количества картинок
    return avg_loss, accuracy

def count_parameters(model):
    """Подсчёт количества обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Оценка на тесте и на трейне
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

results = TrainingResult(
    model_name="fully_conn_model",
    framework="pytorch",
    timestamp=TrainingResult.now(),
    train_accuracy=float(train_accuracy),
    train_loss=float(train_loss),
    test_accuracy=float(test_accuracy),
    test_loss=float(test_loss),
    total_params=count_parameters(model),
    epochs=epochs,
    optimizer="Adam",
    learning_rate=0.001,
)

save_results(results)