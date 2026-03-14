"""
Simple example: Hyperparameter optimization for a neural network.

This example demonstrates the basic usage of AutoForge for optimizing
hyperparameters of a simple neural network on a synthetic dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mltune import Config, Tuner


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Create synthetic dataset
def create_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )

    return train_dataset, val_dataset


# Define objective function
def objective(trial):
    """Objective function to optimize."""
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Create model
    model = SimpleNet(
        input_dim=20,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Create data loaders
    train_dataset, val_dataset = create_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        trial.report(accuracy, step=epoch)

    return accuracy


def main():
    """Run the optimization."""
    print("=" * 60)
    print("AutoForge Example: Hyperparameter Optimization")
    print("=" * 60)

    # Create configuration
    config = Config.from_dict({
        "experiment": {
            "name": "simple_classification",
            "task": "classification",
            "objective": "val_accuracy",
            "direction": "maximize",
        },
        "tuning": {
            "strategy": "bayesian",
            "n_trials": 30,
        },
    })

    # Create tuner
    tuner = Tuner(config, verbose=True)

    # Run optimization
    study = tuner.optimize(objective, n_trials=30)

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Parameter importance
    print("\nParameter Importance:")
    importance = study.param_importance()
    for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {score:.3f}")

    # Save results
    study.save("simple_classification_study.json")
    print("\nStudy saved to: simple_classification_study.json")


if __name__ == "__main__":
    main()
