import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from joblib import Parallel, delayed

from src.dataset import load_and_preprocess_data
from src.config import (
    output_dim,
    hidden_layers,
    learning_rate,
    epochs,
    lambda_values,
    data_path,
)
from src.models.ffnn import FFNN_tensorflow
from src.plots import plot_accuracy_vs_regularization, plot_confusion_matrix

PROJ_ROOT = os.path.join(os.pardir, os.pardir)


def train_and_evaluate(hidden_layer, regularization):
    # Initialize model
    model = FFNN_tensorflow(
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        hidden_layers=hidden_layer,
        regularization=regularization,
        learning_rate=learning_rate,
        cost_function="sparse_cat_cross_entropy",
    )

    # Fit model
    model.fit(X_train, y_train, epochs=epochs)

    # Save the model
    model.model.save(f"models/model_{hidden_layer}_{regularization}.keras")

    # Evaluation
    predictions = model.predict(X_test)
    pred = np.argmax(predictions, axis=1)
    # predict on testset
    predictions_train = model.predict(X_train)
    pred = np.argmax(predictions, axis=1)
    pred_train = np.argmax(predictions_train, axis=1)

    # Calculate accuracy
    test_accuracy = np.mean(pred == y_test)
    train_accuracy = np.mean(pred_train == y_train)

    print(
        f"Accuracy for model with hidden layers {hidden_layer} and regularization {regularization}: {test_accuracy}"
    )
    print(
        f"Train Accuracy for model with hidden layers {hidden_layer} and regularization {regularization}: {train_accuracy}"
    )

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, pred)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels, test_accuracy)
    plt.savefig(f"reports/figures/confusion_matrix_{hidden_layer}_{regularization}.png")

    # Plot accuracy vs regularization

    return test_accuracy, regularization, train_accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_labels = load_and_preprocess_data(
        data_path=data_path
    )

    # Initialize lists to store results
    regularization_values = []
    accuracy_values = []
    accuracy_values_train = []

    # Parallel processing
    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate)(hidden_layers[j], regularization)
        for j in range(len(hidden_layers))
        for regularization in lambda_values
    )

    # Append results to lists
    for test_accuracy, regularization, train_accuracy in results:
        regularization_values.append(regularization)
        accuracy_values.append(test_accuracy)
        accuracy_values_train.append(train_accuracy)

    plot_accuracy_vs_regularization(
        lambda_values, accuracy_values_train, accuracy_values
    )
    plt.savefig(f"reports/figures/accuracy_vs_regularization _{regularization}.png")
