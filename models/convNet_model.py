#  сверточная модель нейросети с использованием  Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.mnist import X_training, X_test, y_training, y_test
from handlers.save_handler import save_results
from handlers.training_result import TrainingResult

# Тензоры и DataLoader
X_training = X_training.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

train_dataset = TensorDataset(
    torch.tensor(X_training),
    torch.tensor(y_training)  # для классификации
)
test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Свёрточная модель
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Блок извлечения признаков (свёртки)
        self.conv_layers = nn.Sequential(
            # Слой 1: 1 канал → 32 фильтра, ядро 3x3
            # 32 фильтра размером 3×3 сканируют изображение. 
            # Каждый фильтр — это матрица весов 3×3, которая ищет свой паттерн
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # уменьшение размера 28x28 → 14x14

            # Слой 2: 32 фильтра → 64 фильтра
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14x14 → 7x7
        )
        # Блок классификации (полносвязные слои)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                  # 64 * 7 * 7 = 3136 нейронов
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)             # 10 классов
        )

    def forward(self, x):
        x = self.conv_layers(x)   # извлечение признаков
        x = self.fc_layers(x)     # классификация
        return x

# Обучение
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

#  Оценка и сбор метрик

def evaluate_model(model, loader, criterion, device):
    """Вычисляет accuracy и loss на переданном loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def count_parameters(model):
    """Подсчёт количества обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Вычисляем метрики
train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

results = TrainingResult(
    model_name="convNet_model",
    framework="pytorch",
    timestamp=TrainingResult.now(),
    train_accuracy=float(train_acc),
    train_loss=float(train_loss),
    test_accuracy=float(test_acc),
    test_loss=float(test_loss),
    total_params=count_parameters(model),
    epochs=epochs,
    optimizer="Adam",
    learning_rate=0.001,
)

save_results(results)