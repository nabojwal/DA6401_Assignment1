
import numpy as np
import wandb
from src.ann.neural_layer import NeuralLayer
from src.ann.activations import ReLU, Sigmoid, Tanh
from src.ann.objective_functions import CrossEntropyLoss, MSELoss
from src.ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam

class NeuralNetwork:
    """
    Main model class that orchestrates training and inference.
    """

    def __init__(self, cli_args):
        self.input_dim = 784
        self.output_dim = 10
        
        if isinstance(cli_args.hidden_size, int):
            self.hidden_layers = [cli_args.hidden_size] * cli_args.num_layers
        else:
            self.hidden_layers = cli_args.hidden_size

        self.activation_name = cli_args.activation
        self.loss_name = cli_args.loss
        self.weight_init = cli_args.weight_init
        self.learning_rate = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        self.optimizer_name = cli_args.optimizer

        self.layers = []
        self.activations = []

        if self.activation_name == "relu":
            activation_class = ReLU
        elif self.activation_name == "sigmoid":
            activation_class = Sigmoid
        elif self.activation_name == "tanh":
            activation_class = Tanh
        else:
            raise ValueError("Unsupported activation")

 
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_layers:
            layer = NeuralLayer(prev_dim, hidden_dim, weight_init=self.weight_init)
            self.layers.append(layer)
            self.activations.append(activation_class())
            prev_dim = hidden_dim

        final_layer = NeuralLayer(prev_dim, self.output_dim, weight_init=self.weight_init)
        self.layers.append(final_layer)

        if self.loss_name == "cross_entropy":
            self.loss_function = CrossEntropyLoss()
        elif self.loss_name == "mse":
            self.loss_function = MSELoss()
        else:
            raise ValueError("Unsupported loss")
        
        opt_name = cli_args.optimizer.lower()

        if opt_name == "sgd":
            self.optimizer = SGD(self.layers, self.learning_rate)

        elif opt_name == "momentum":
            self.optimizer = Momentum(self.layers, self.learning_rate)

        elif opt_name == "nag":
            self.optimizer = NAG(self.layers, self.learning_rate)

        elif opt_name == "rmsprop":
            self.optimizer = RMSProp(self.layers, self.learning_rate)

        elif opt_name == "adam":
            self.optimizer = Adam(self.layers, self.learning_rate)

        elif opt_name == "nadam":
            self.optimizer = Nadam(self.layers, self.learning_rate)

        else:
            raise ValueError(f"Unsupported optimizer: {cli_args.optimizer}")

    def forward(self, X):

        out = X

        for i in range(len(self.layers) - 1):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)
            # activation dist.for 1st hidden layer
            # if i == 0:
            #     wandb.log({
            #         "layer1_activation_mean": np.mean(out),
            #         "layer1_activation_zero_fraction": np.mean(out == 0)
            #     })
        logits = self.layers[-1].forward(out)

        return logits

    def backward(self, y_true, logits):
        loss = self.loss_function.forward(logits, y_true)
        grad = self.loss_function.backward()
        grad = self.layers[-1].backward(grad, self.weight_decay)
        for i in reversed(range(len(self.layers) - 1)):
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad, self.weight_decay)

        return loss  

    def update_weights(self):

        self.optimizer.step()

    def train(self, X_train, y_train, epochs, batch_size):

        history = {
            "train_loss": [],
            "train_accuracy": []
        }

        N = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            total_loss = 0

            for i in range(0, N, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                logits = self.forward(X_batch)
                loss= self.backward(y_batch, logits)

                # # gradient norm of first hidden layer
                # grad_norm = np.linalg.norm(self.layers[0].grad_W)

                # wandb.log({
                #     "grad_norm_layer1": grad_norm
                # })
                # g = self.layers[0].grad_W
                # wandb.log({
                #     "grad_neuron_1": g[0,0],
                #     "grad_neuron_2": g[0,1],
                #     "grad_neuron_3": g[0,2],
                #     "grad_neuron_4": g[0,3],
                #     "grad_neuron_5": g[0,4]
                # })
                self.update_weights()

                total_loss += loss

            avg_loss = total_loss / (N // batch_size)

            _, acc = self.evaluate(X_train, y_train)

            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(acc)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

        return history


    def evaluate(self, X, y):

        logits = self.forward(X)

        loss = self.loss_function.forward(logits, y)

        predictions = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == labels)

        return loss, accuracy
    