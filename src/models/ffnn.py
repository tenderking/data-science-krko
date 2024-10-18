from tensorflow.data import Dataset, AUTOTUNE  # type: ignore

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Input
from keras._tf_keras.keras.optimizers import SGD
from keras._tf_keras.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import EarlyStopping


class FFNN_tensorflow:
    """
    Feedforward Neural Network (FFNN) implemented using TensorFlow.

    Parameters:
    - input_dim (int): Dimensionality of the input features.
    - output_dim (int): Dimensionality of the output (number of classes).
    - hidden_layers (tuple): Tuple specifying the sizes of hidden layers.
    - hidden_activation (str): Activation function for hidden layers.
    - output_activation (str): Activation function for the output layer.
    - cost_function (str): Loss function to optimize during training.
    - learning_rate (float): Learning rate for the stochastic gradient descent optimizer.
    - regularization (str): Regularization method ('l1', 'l2', or None).

    Methods:
    - __init__: Initializes the FFNN model with the specified parameters.
    - build_model: Constructs and compiles the FFNN model architecture.
    - fit: Trains the FFNN model on the provided data.
    - predict: Generates predictions for input data using the trained model.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers=(64, 32),
        hidden_activation="relu",
        output_activation="softmax",
        cost_function="mean_squared_error",
        learning_rate=0.01,
        regularization=None,
        patience=10,
    ):
        """
        Initialize the FFNN model with specified parameters.

        Parameters:
        - input_dim (int): Dimensionality of the input features.
        - output_dim (int): Dimensionality of the output (number of classes).
        - hidden_layers (tuple): Tuple specifying the sizes of hidden layers.
        - hidden_activation (str): Activation function for hidden layers.
        - output_activation (str): Activation function for the output layer.
        - cost_function (str): Loss function to optimize during training.
        - learning_rate (float): Learning rate for the stochastic gradient descent optimizer.
        - regularization (str): Regularization method ('l1', 'l2', or None).
        """

        # Parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.model = self.build_model()
        self.patience = patience

    def build_model(self):
        """
        Build and compile the FFNN model architecture.

        Returns:
        - tf.keras.Model: Compiled FFNN model.
        """

        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))  # Define input shape with Input layer
        model.add(
            Dense(
                self.hidden_layers[0],
                activation="relu",
                kernel_regularizer=l2(self.regularization),
            )
        )

        for layer_size in self.hidden_layers[1:]:
            model.add(Dense(layer_size, activation=self.hidden_activation))

        model.add(Dense(self.output_dim, activation=self.output_activation))

        # Gradient method for updating SGD
        optimizer = SGD(learning_rate=self.learning_rate)

        # Loss function
        if self.cost_function == "sparse_cat_cross_entropy":
            loss_function = SparseCategoricalCrossentropy()
        elif self.cost_function == "mean_squared_error":
            loss_function = MeanSquaredError()
        else:
            raise ValueError(
                "Invalid cost_function. Choose 'sparse_cat_cross_entropy' or 'mean_squared_error'"
            )

        model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=[
                "accuracy",
            ],
        )

        return model

    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None):
        """
        Train the FFNN model on the provided data.

        Parameters:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): Target labels.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - validation_data (tuple): Validation data as (X_val, y_val).
        """

        # Create a tf.data.Dataset object
        dataset = Dataset.from_tensor_slices((X, y))

        # Apply batching and prefetching
        dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            patience=self.patience,
            # Stop after 'patience' epochs with no improvement
            restore_best_weights=True,  # Restore weights from the epoch with the best val_loss
        )

        # Train the model using the dataset and early stopping
        # Train the model using the dataset and early stopping
        history = self.model.fit(  # Assign the result to history
            dataset,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[early_stopping],  # Add the early stopping callback
            verbose=1,
        )
        return history

    def predict(self, X):
        """
        Generate predictions for input data using the trained model.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Model predictions.
        """

        return self.model.predict(X)
