# src/plots.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_labels, accuracy):
    """Plots a confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Accuracy: {accuracy:.4f}")
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.show()

def plot_accuracy_vs_regularization(regularization_values, accuracy_values_train, accuracy_values):
    """Plots accuracy against regularization parameter."""
    plt.plot(regularization_values, accuracy_values_train, marker='o', label="y_train")
    plt.plot(regularization_values, accuracy_values, marker='o', label="y_test")
    plt.xlabel('Regularization Parameter')
    plt.ylabel('Accuracy')
    plt.title('Effect of Regularization Parameter on Test and Train Set Accuracy')
    plt.xscale('log') 
    plt.show()
    print("plot_accuracy_vs_regularization")