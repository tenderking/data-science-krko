# src/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(cm, class_labels, accuracy):
    """Plots a confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Accuracy: {accuracy:.4f}")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )


def plot_scatterplot(x: list[int], xlabel: str, y: list[float], ylabel: str):
    """Plots a scatterplot."""
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"{xlabel} vs {ylabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def plot_accuracy_vs_epoch(history):
    """Plots accuracy vs epoch."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()


def plot_loss_vs_epoch(history):
    """Plots loss vs epoch."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()


def plot_learning_rate_vs_accuracy(learning_rates, accuracies):
    """Plots learning rate vs accuracy."""
    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates, accuracies, marker="o")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Learning Rate vs. Accuracy")
    plt.xscale("log")  # Use logarithmic scale for learning rate


def plot_feature_importance(importances, indices, feature_names):
    """Plots feature importance."""
    # Plot the feature importances
    plt.figure(figsize=(13, 16))
    plt.bar(range(6), importances[indices], align="center")
    plt.xticks(range(6), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Variable Importance - Decision Tree")

    folder_path = "reports/figures/"

    # Specifying the file name  and join it with the folder path
    file_path = os.path.join(
        folder_path, f" DecisionTreeClassifier_variable_{feature_names}.png"
    )

    # Save the plot
    plt.savefig(file_path)
    return file_path
