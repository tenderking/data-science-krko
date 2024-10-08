
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os

from src.dataset import load_and_preprocess_data
from src.config import output_dim, hidden_layers, learning_rate, epochs, lambda_values, data_path
from src.models.ffnn import FFNN_tensorflow
from src.plots import plot_accuracy_vs_regularization, plot_confusion_matrix
PROJ_ROOT = os.path.join(os.pardir, os.pardir)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, class_labels = load_and_preprocess_data(data_path = data_path)

    for j in range(len(hidden_layers)):  # Iterate over different hidden layer configurations
        for regularization in lambda_values:
            # Initialized models
            model = FFNN_tensorflow(input_dim=X_train.shape[1], output_dim=output_dim, 
                                     hidden_layers=hidden_layers[j],
                                     regularization=regularization, learning_rate=learning_rate,
                                     cost_function="sparse_cat_cross_entropy")

            # Fit model
            model.fit(X_train, y_train, epochs=epochs)

            # Save the model
            model.model.save(f"models/model_{j}_{regularization}.h5")  # Include j in the filename

            # Evaluation
            predictions = model.predict(X_test)
            pred = np.argmax(predictions, axis=1)  # Get predicted classes

            # Calculate and print accuracy (optional)
            test_accuracy = np.mean(pred == y_test)
            print(f"Accuracy for model {j} with regularization {regularization}: {test_accuracy}")

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, pred)

            # Plot confusion matrix
            plot_confusion_matrix(cm, class_labels, test_accuracy)
            plt.savefig(f"reports/figures/confusion_matrix_{j}_{regularization}.png")

            # Plot accuracy vs regularization
            plot_accuracy_vs_regularization(lambda_values, model.accuracy_values_train, model.accuracy_values)
            plt.savefig(f"reports/figures/accuracy_vs_regularization_{j}_{regularization}.png")