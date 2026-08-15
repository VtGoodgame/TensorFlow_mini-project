import numpy as np

# Отключаем научную нотацию — числа будут читаемыми
np.set_printoptions(suppress=True, precision=4)

# 4 признака: длина чашелистика, ширина чашелистика,
# длина лепестка, ширина лепестка
X = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Цветок 1: setosa
    [7.0, 3.2, 4.7, 1.4],  # Цветок 2: versicolor
    [6.3, 3.3, 6.0, 2.5],  # Цветок 3: virginica
    [4.9, 3.0, 1.4, 0.2],  # Цветок 4: setosa
])

# Метки: 0=setosa, 1=versicolor, 2=virginica
# В one-hot формате (каждая метка — вектор из 3 чисел)
iris_names = ("setosa", "versicolor", "virginica")
y = np.array([
    [1, 0, 0],  # setosa
    [0, 1, 0],  # versicolor
    [0, 0, 1],  # virginica
    [1, 0, 0],  # setosa
])

print("Входные данные X (4 цветка x 4 признака):")
print(X)
print("\nМетки y в one-hot формате:")
print(y)
print(f"Форма X: {X.shape}")  # (4, 4)
print(f"Форма y: {y.shape}")  # (4, 3)

# Фиксируем random seed, чтобы результаты были одинаковыми при каждом запуске
# Компьютер не умеет генерировать настоящие случайные числа. 
# Он использует псевдослучайный алгоритм — длинную математическую последовательность,
# которая выглядит как случайная.

# seed (зерно) — это начальная точка в этой последовательности.
# Одно и то же зерно → одна и та же последовательность.
np.random.seed(42)

# Слой 1: 4 входа → 8 нейронов
W1 = np.random.randn(4, 8) * 0.5  # 4×8 = 32 веса
b1 = np.zeros((1, 8))              # 8 смещений

# Слой 2: 8 нейронов → 3 выхода
W2 = np.random.randn(8, 3) * 0.5  # 8×3 = 24 веса
b2 = np.zeros((1, 3))              # 3 смещения

# Смещение — это базовая линия, от которой нейрон отталкивается.
print("Веса слоя 1 W1 (4x8):")
print(W1)
print(f"\nФорма W1: {W1.shape} — 32 параметра")

print("\nСмещения слоя 1 b1 (1x8):")
print(b1)
# Вес это коэффициент важности данного параметра, 
# изначально этот вес рандомный, позже вес меняется из обучения. 
print("\nВеса слоя 2 W2 (8x3):")
print(W2)
print(f"\nФорма W2: {W2.shape} — 24 параметра")

print(f"\nВсего параметров: {W1.size + b1.size + W2.size + b2.size}")

# # ШАГ 3: Прямой проход (Forward Pass)
# # Слой 1
print("\n--- Скрытый слой ---")
print("1. Умножаем входы на веса и добавляем смещения:")
z1 = np.dot(X, W1) + b1  # (4×4) · (4×8) + (1×8) = (4×8)
print(f"   z1 = X @ W1 + b1, форма: {z1.shape}")
print("   Значения z1 (до активации):")
print(z1)

print("\n2. Применяем ReLU (всё отрицательное -> 0):")
a1 = np.maximum(0, z1)  # ReLU
print("   a1 = relu(z1):")
print(a1)

# Слой 2 (выходной)
print("\n--- Выходной слой ---")
print("3. Умножаем выходы скрытого слоя на веса и добавляем смещения:")
z2 = np.dot(a1, W2) + b2  # (4×8) · (8×3) + (1×3) = (4×3)
print(f"   z2 = a1 @ W2 + b2, форма: {z2.shape}")
print("   Значения z2 (до Softmax):")
print(z2)

print("\n4. Применяем Softmax (превращаем в вероятности):")
def softmax(x):
    """Softmax для каждой строки матрицы."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # вычитаем max для стабильности
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

y_pred = softmax(z2)
print("   y_pred (предсказания):")
print(y_pred)
print("\n   Сумма вероятностей в каждой строке:")
print(np.sum(y_pred, axis=1))  # Должна быть 1.0

# ШАГ 4: Вычисление ошибки (Loss)
print("\n" + "=" * 60)
print("ШАГ 4: ВЫЧИСЛЕНИЕ ОШИБКИ")
print("=" * 60)

# Categorical Crossentropy
# loss = -1/N * sum(y_true * ln(y_pred))
# Добавляем маленькое число 1e-15, чтобы ln(0) не взорвался
epsilon = 1e-15
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
loss = -np.mean(np.sum(y * np.log(y_pred_clipped), axis=1))

print(f"Правильные ответы y:")
print(y)
print(f"\nПредсказания y_pred:")
print(y_pred)
print(f"\nLoss (ошибка): {loss:.4f}")

# Посчитаем loss для каждого цветка отдельно
print("\nОшибка для каждого цветка:")
individual_losses = -np.sum(y * np.log(y_pred_clipped), axis=1)
for i in range(4):
    true_class = np.argmax(y[i])
    pred_class = np.argmax(y_pred[i])
    print(f"  Цветок {i+1}: истинный класс={true_class}, "
          f"предсказанный={pred_class}, loss={individual_losses[i]:.4f}")

# ШАГ 5: Обратный проход (Backpropagation)
print("\n" + "=" * 60)
print("ШАГ 5: ОБРАТНЫЙ ПРОХОД — СЧИТАЕМ ГРАДИЕНТЫ")
print("=" * 60)

print("Градиент = насколько сильно надо изменить параметр, чтобы уменьшить ошибку")
print("Положительный градиент -> уменьшаем параметр")
print("Отрицательный градиент -> увеличиваем параметр\n")

# Количество примеров
m = X.shape[0]  # 4

# Градиенты выходного слоя
# dL/dz2 = y_pred - y_true  (это производная Softmax + Crossentropy вместе)
dz2 = y_pred - y  # (4×3)
print("Градиент выходного слоя dz2 = y_pred - y:")
print(dz2)
print(f"   Форма: {dz2.shape}")

# Градиенты весов и смещений выходного слоя
dW2 = np.dot(a1.T, dz2) / m  # (8×4) · (4×3) = (8×3)
db2 = np.sum(dz2, axis=0, keepdims=True) / m  # (1×3)

print("\nГрадиенты весов выходного слоя dW2 (8x3):")
print(dW2)
print(f"   Средняя величина градиента: {np.mean(np.abs(dW2)):.6f}")

# Градиенты скрытого слоя
# Производная ReLU: если a1 > 0 → 1, иначе → 0
da1 = np.dot(dz2, W2.T)  # (4×3) · (3×8) = (4×8)
dz1 = da1 * (a1 > 0).astype(float)  # умножаем на производную ReLU

dW1 = np.dot(X.T, dz1) / m  # (4×4) · (4×8) = (4×8)
db1 = np.sum(dz1, axis=0, keepdims=True) / m  # (1×8)

print("\nГрадиенты весов скрытого слоя dW1 (4x8):")
print(dW1)
print(f"   Средняя величина градиента: {np.mean(np.abs(dW1)):.6f}")

# ШАГ 6: Обновление весов (один шаг оптимизатора)
print("\n" + "=" * 60)
print("ШАГ 6: ОБНОВЛЕНИЕ ВЕСОВ")
print("=" * 60)

learning_rate = 0.1

print(f"Learning rate: {learning_rate}")
print("\nФормула: новый_вес = старый_вес - learning_rate x градиент\n")

# Сохраняем старые веса для сравнения
W1_old = W1.copy()
W2_old = W2.copy()

# Обновляем
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1

# Показываем изменения на примере первых 3 весов слоя 2
print("Пример обновления (первые 3 веса W2):")
for i in range(3):
    old_val = W2_old[0, i]
    grad = dW2[0, i]
    new_val = W2[0, i]
    change = new_val - old_val
    print(f"  W2[0,{i}]: {old_val:+.4f} - {learning_rate} x ({grad:+.4f}) = {new_val:+.4f}  (изменение: {change:+.4f})")

print(f"\nМаксимальное изменение весов W1: {np.max(np.abs(W1 - W1_old)):.6f}")
print(f"Максимальное изменение весов W2: {np.max(np.abs(W2 - W2_old)):.6f}")

# ШАГ 7: Проверка — стал ли loss меньше?
print("\n" + "=" * 60)
print("ШАГ 7: ПРОВЕРКА ПОСЛЕ ОДНОГО ШАГА")
print("=" * 60)

# Прямой проход с новыми весами
z1_new = np.dot(X, W1) + b1
a1_new = np.maximum(0, z1_new)
z2_new = np.dot(a1_new, W2) + b2
y_pred_new = softmax(z2_new)

# Новый loss
y_pred_new_clipped = np.clip(y_pred_new, epsilon, 1 - epsilon)
loss_new = -np.mean(np.sum(y * np.log(y_pred_new_clipped), axis=1))

# Сравниваем accuracy до и после
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

acc_before = accuracy(y, y_pred)
acc_after = accuracy(y, y_pred_new)

print(f"Loss ДО обновления:     {loss:.4f}")
print(f"Loss ПОСЛЕ обновления:  {loss_new:.4f}")
print(f"Loss уменьшился на:     {loss - loss_new:.4f}")
print(f"Accuracy ДО:   {acc_before:.0%}")
print(f"Accuracy ПОСЛЕ: {acc_after:.0%}")


# ШАГ 8: А теперь 500 итераций обучения
print("\n" + "=" * 60)
print("ШАГ 8: ПОЛНОЕ ОБУЧЕНИЕ (500 итераций)")
print("=" * 60)
# Переинициализируем веса
np.random.seed(42)
W1 = np.random.randn(4, 8) * 0.5
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 3) * 0.5
b2 = np.zeros((1, 3))

learning_rate = 0.1
losses = []
for epoch in range(500):
    # Прямой проход
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)
    
    # Loss
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y * np.log(y_pred_clipped), axis=1))
    losses.append(loss)
    
    # Обратный проход
    dz2 = y_pred - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (a1 > 0).astype(float)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    # Обновление
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    # Вывод каждые 100 эпох
    if epoch % 100 == 0:
        acc = accuracy(y, y_pred)
        print(f"Эпоха {epoch:3d}: loss={loss:.4f}, accuracy={acc:.0%}")

# Финальный результат
print(f"\nФинальный loss: {losses[-1]:.4f}")
print(f"Начальный loss: {losses[0]:.4f}")
print(f"Loss уменьшился в {losses[0]/losses[-1]:.1f} раз!")

# Финальные предсказания
print("\nФинальные предсказания:")
for i in range(4):
    true_class = np.argmax(y[i])
    pred_class = np.argmax(y_pred[i])
    confidence = y_pred[i][pred_class]
    status = "[OK]" if true_class == pred_class else "[FAIL]"
    print(f"  Цветок {i+1}: истинный={iris_names[true_class]}, "
          f"предсказанный={iris_names[pred_class]}, "
          f"уверенность={confidence:.2%} {status}")

# График loss'а (текстовый, без matplotlib)
print("\nГрафик уменьшения ошибки (текстовый):")
print("Эпоха   Loss")
for i in [0, 50, 100, 200, 300, 400, 499]:
    bar = "#" * int((1 - losses[i]/losses[0]) * 50)
    print(f"{i:5d}: {losses[i]:.4f} {bar}")