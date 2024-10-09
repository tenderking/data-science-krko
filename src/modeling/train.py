import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from joblib import Parallel, delayed
import wandb  # Import wandb


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

wandb.login()  # Login to your WandB account

PROJ_ROOT = os.path.join(os.pardir, os.pardir)


# Initialize WandB with config
def train_and_evaluate(hidden_layer, learning_rate_i):
    # Initialize WandB with a config dictionary
    run = wandb.init(
        project="my-awesome-project",  # Replace with your actual project name
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "hidden_layers": hidden_layers,
            "lambda_values": lambda_values,
        },
        group="my_experiment_group",
    )

    assert run is wandb.run
    # Initialize model
    model = FFNN_tensorflow(
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        hidden_layers=hidden_layer,
        regularization=lambda_values,
        learning_rate=learning_rate_i,
        cost_function="sparse_cat_cross_entropy",
    )

    # Fit model
    history = model.fit(X_train, y_train, epochs=epochs)
    # Store the history in the model object

    # Save the model
    model_path = f"models/model_{str(hidden_layer).replace('(', '').replace(')', '').replace(', ', '_')}_{learning_rate_i}.keras"
    model.model.save(model_path)

    # Evaluation
    predictions = model.predict(X_test)
    pred = np.argmax(predictions, axis=1)
    predictions_train = model.predict(X_train)
    pred_train = np.argmax(predictions_train, axis=1)

    # Calculate accuracy
    test_accuracy = np.mean(pred == y_test)
    train_accuracy = np.mean(pred_train == y_train)

    print(
        f"Accuracy for model with hidden layers {hidden_layer} and learning rate {learning_rate_i}: {test_accuracy}"
    )
    print(
        f"Train Accuracy for model with hidden layers {hidden_layer} and learning rate {learning_rate_i}: {train_accuracy}"
    )

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, pred)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels, test_accuracy)
    confusion_matrix_path = f"reports/figures/confusion_matrix_{str(hidden_layer).replace('(', '').replace(')', '').replace(', ', '_')}_{learning_rate_i}.png"
    plt.savefig(confusion_matrix_path)

    # Save the model
    model_path = f"models/model_{str(hidden_layer).replace('(', '').replace(')', '').replace(', ', '_')}_{learning_rate_i}.keras"
    model.model.save(model_path)

    # Log to WandB
    artifact_name = f"model_{str(hidden_layer).replace('(', '').replace(')', '').replace(', ', '_')}_{str(learning_rate_i).replace('.', '_')}"
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(model_path)  # Add the model file to the artifact

    wandb.log_artifact(artifact)  # Log the artifact

    wandb.log(
        {
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
            "confusion_matrix": wandb.Image(confusion_matrix_path),
            "hidden_layers": hidden_layer,
            "accuracy": history.history["accuracy"],
            "epoch": list(range(epochs)),
            "loss": history.history["loss"],
            "learning_rate": learning_rate_i,
        }
    )
    wandb.finish()
    return test_accuracy, learning_rate_i, train_accuracy


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, class_labels = load_and_preprocess_data(
        data_path=data_path
    )

    # Initialize lists to store results
    learning_rate_values = []
    accuracy_values = []
    accuracy_values_train = []

    # Parallel processing
    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate)(hidden_layers[j], rate)
        for j in range(len(hidden_layers))
        for rate in learning_rate
    )

    # Append results to lists
    # Append results to lists
    for (
        test_accuracy,
        learning_rate_i,
        train_accuracy,
    ) in results:  # Extract learning_rate_i
        learning_rate_values.append(learning_rate_i)  # Store learning rates
        accuracy_values.append(test_accuracy)
        accuracy_values_train.append(train_accuracy)

    if len(learning_rate_values) != len(accuracy_values_train) or len(
        learning_rate_values
    ) != len(accuracy_values):
        print(f"Length of regularization_values: {len(learning_rate_values)}")
        print(f"Length of accuracy_values_train: {len(accuracy_values_train)}")
        print(f"Length of accuracy_values: {len(accuracy_values)}")
    else:
        # Plot accuracy vs regularization
        plot_accuracy_vs_regularization(
            lambda_values, accuracy_values_train, accuracy_values
        )
        accuracy_vs_learnng_rate_path = "reports/figures/accuracy_vs_learning_rate.png"
        plt.savefig(accuracy_vs_learnng_rate_path)
        wandb.init(project="my-awesome-project", group="my_experiment_group")

        # Log the final plot to WandB
        wandb.log(
            {"accuracy_vs_learning_rate": wandb.Image(accuracy_vs_learnng_rate_path)}
        )
        wandb.finish()
