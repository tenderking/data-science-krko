from src.models.random_search import RandomSearch
from src.dataset import load_and_preprocess_data
from src.config import data_path
from src.plots import plot_feature_importance
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

wandb.login()


def main():
    # Load data
    wandb.init(
        project="my-awesome-project",
        group="random_search",
    )

    X_train, X_test, y_train, y_test, class_labels = load_and_preprocess_data(
        data_path=data_path
    )

    # Initialize RandomSearch object
    random_search = RandomSearch(X_train, y_train, X_test, y_test)

    # Build models
    random_search.build_model()

    # Fit models and log results to wandb
    results, importances, feature_names, indices, models = random_search.fit(
        X_train, y_train
    )

    for model_name, result_df in results.items():
        # Log the results DataFrame as a table
        wandb.log({f"{model_name}_results": wandb.Table(dataframe=result_df)})

        # Log the best model's accuracy
        best_accuracy = result_df["Accuracy"].max()
        wandb.log({f"{model_name}_best_accuracy": best_accuracy})

        # Log feature importances for the best model
        if (
            model_name != "Bagging"
        ):  # Assuming 'Bagging' doesn't have direct feature importances
            feature_importance_plot = plot_feature_importance(
                importances, indices, feature_names
            )
            wandb.log(
                {
                    f"{model_name}_feature_importance": wandb.Image(
                        feature_importance_plot
                    )
                }
            )

            # Log confusion matrix for the best model
            best_model = models[model_name]
            y_pred = best_model.predict(X_test)
            print(y_pred)
            try:
              cm = confusion_matrix(y_test, y_pred, labels=class_labels)
              fig, ax = plt.subplots()
              sns.heatmap(cm, annot=True, fmt="d", ax=ax)
              ax.set_xlabel("Predicted labels")
              ax.set_ylabel("True labels")
              ax.set_title("Confusion Matrix")
              wandb.log({f"{model_name}_confusion_matrix": wandb.Image(fig)})
            except Exception as e:
              print(f"Failed to log confusion matrix for {model_name}: {e}")
              print(cm)

    wandb.finish()


if __name__ == "__main__":
    main()
