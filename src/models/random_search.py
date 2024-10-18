from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# from sklearn.metrics import make_scorer, matthews_corrcoef
from src.config import param_dist_xgb, param_dist_rf, param_dist_dt, param_dist_bagging


class RandomSearch:
    """
    Builds and evaluates multiple machine learning models using RandomizedSearchCV.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the ModelBuilder with training and testing data.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}  # To store initialized models

    def build_model(self):
        """
        Instantiates the machine learning models with initial parameters.
        """
        self.models["XGBoost"] = xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", random_state=42
        )
        self.models["Random Forest"] = RandomForestClassifier(random_state=42)
        self.models["Decision Tree"] = DecisionTreeClassifier(random_state=42)
        self.models["Bagging"] = BaggingClassifier(
            estimator=self.models["Decision Tree"], random_state=42
        )

        # Perform RandomizedSearchCV with verbose output
        self.random_searches = {
            "XGBoost": RandomizedSearchCV(
                self.models["XGBoost"],
                param_distributions=param_dist_xgb,
                n_iter=20,
                cv=5,
                scoring="f1_micro",
                verbose=2,  # Increased verbosity
            ),
            "Random Forest": RandomizedSearchCV(
                self.models["Random Forest"],
                param_distributions=param_dist_rf,
                n_iter=20,
                cv=5,
                scoring="f1_micro",
                verbose=2,  # Increased verbosity
            ),
            "Decision Tree": RandomizedSearchCV(
                self.models["Decision Tree"],
                param_distributions=param_dist_dt,
                n_iter=20,
                cv=5,
                scoring="f1_micro",
                verbose=2,  # Increased verbosity
            ),
            "Bagging": RandomizedSearchCV(
                self.models["Bagging"],
                param_distributions=param_dist_bagging,
                n_iter=20,
                cv=5,
                scoring="f1_micro",
                verbose=2,  # Increased verbosity
            ),
        }

    def fit(self, X, y):
        """
        Fits the models, tunes hyperparameters, and evaluates performance.
        """

        results = {}
        for model_name, search in self.random_searches.items():
            print(f"Fitting {model_name}...")

            if model_name == "XGBoost":
                eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

                # Define a callback function (must inherit from xgboost.callback.TrainingCallback)
                class LogEvaluation(
                    xgb.callback.TrainingCallback
                ):  # Inherit from TrainingCallback
                    def after_iteration(self, model, epoch, evals_log):
                        print(f"Boosting round {epoch}: {evals_log}")
                        return False  # Return False to continue training

                # Modify the XGBoost estimator within RandomizedSearchCV
                search.estimator.set_params(
                    eval_set=eval_set,
                    callbacks=[LogEvaluation()],  # Create an instance of the callback
                )

                search.fit(self.X_train, self.y_train)  # No need for fit_params here

            else:
                search.fit(self.X_train, self.y_train)

            hyperparameters_list = search.cv_results_["params"]
            accuracies = []
            for i in range(len(hyperparameters_list)):
                hyperparameters = hyperparameters_list[i].copy()
                model = self.models[model_name].set_params(**hyperparameters)
                model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, predictions)
                cm = confusion_matrix(self.y_test, predictions)
                hyperparameters["Accuracy"] = accuracy
                hyperparameters["Confusion Matrix"] = cm
                accuracies.append(hyperparameters)

            results[model_name] = pd.DataFrame(accuracies)

            # Get the best model with tuned hyperparameters
            best_hyperparameters = search.best_params_
            best_model = self.models[model_name].set_params(**best_hyperparameters)
            best_model.fit(X, y)  # Fit on the entire dataset (X, y)

            # Calculate feature importances and indices
            if model_name == "Bagging":  # Special handling for BaggingClassifier
                all_importances = []
                for est in best_model.estimators_:
                    all_importances.append(est.feature_importances_)
                importances = np.mean(all_importances, axis=0)

                # Calculate indices for BaggingClassifier
                indices = np.argsort(importances)[::-1]

            else:  # For models with direct feature_importances_ attribute
                importances = best_model.feature_importances_
                indices = np.argsort(importances)[::-1]

            feature_names = self.X_train.columns

            self.models[model_name] = best_model  # Store the best model

        return results, importances, feature_names, indices, self.models
