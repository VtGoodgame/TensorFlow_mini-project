"""Единая схема результатов обучения модели."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TrainingResult:
    """Результаты обучения модели.

    Единый набор полей для всех моделей проекта. Неприменимые
    для конкретного фреймворка поля остаются со значением None.
    """

    model_name: str
    framework: str
    timestamp: str
    train_accuracy: Optional[float] = None
    train_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    total_params: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None

    @staticmethod
    def now() -> str:
        """Текущая метка времени для поля timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        """Словарь с полным набором ключей в едином порядке."""
        return {
            "model_name": self.model_name,
            "framework": self.framework,
            "timestamp": self.timestamp,
            "train_accuracy": self.train_accuracy,
            "train_loss": self.train_loss,
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
            "val_accuracy": self.val_accuracy,
            "val_loss": self.val_loss,
            "total_params": self.total_params,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
        }
