

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, SparseCategoricalCrossentropy # type: ignore
from sklearn.metrics import r2_score
from tensorflow.keras import regularizers, Input # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

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

    def __init__(self, input_dim, output_dim, hidden_layers=(64, 32),
                 hidden_activation='relu', output_activation='softmax',
                 cost_function='mean_squared_error', learning_rate=0.01,
                 regularization=None):
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

    def build_model(self):
        """
        Build and compile the FFNN model architecture.

        Returns:
        - tf.keras.Model: Compiled FFNN model.
        """

        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))  # Define input shape with Input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', kernel_regularizer=l2(self.regularization)))


        for layer_size in self.hidden_layers[1:]:
            model.add(Dense(layer_size, activation=self.hidden_activation))

        model.add(Dense(self.output_dim, activation=self.output_activation))

        # Gradient method for updating SGD
        optimizer = SGD(learning_rate=self.learning_rate)

        # Loss function
        if self.cost_function == 'sparse_cat_cross_entropy':
            loss_function = SparseCategoricalCrossentropy()

        # Regularizer
        if self.regularization == 'l1':
            regularizer = regularizer.l1()
        elif self.regularization == 'l2':
            regularizer = regularizer.l2()
        else:
            regularizer = None

        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'], loss_weights=regularizer)

        return model

    def fit(self, X, y, epochs=100, batch_size=None, validation_data=None):
        """
        Train the FFNN model on the provided data.

        Parameters:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): Target labels.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - validation_data (tuple): Validation data as (X_val, y_val).
        """

        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, X):
        """
        Generate predictions for input data using the trained model.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Model predictions.
        """

        return self.model.predict(X)

