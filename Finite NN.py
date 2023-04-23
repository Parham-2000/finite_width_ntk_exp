### **KERAS** is a major, user-friendly framework for doing deep learning. We will be experimenting with it here


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.optimizers import SGD
import matplotlib.pyplot as plt

normalize_raw_data = True
cheat_and_use_future_data = False

split = int(signals.shape[0] / 2)
train_labels = labels[:split]
test_labels = labels[split:]
# Train and test datasets
X_test = signals[split:]
X_train = signals[:split]
y_test = test_labels
y_train = train_labels


# We will be using a very simple, sequential NN architecture. See, [ResNet in TensorFlow](https://subscription.packtpub.com/book/data/9781838829131/2/ch02lvl1sec14/implementing-resnet-from-scratch) for more complex architectures

# here we use the remarkably simple Keras framework to quickly build
# any NN architecture we want


# costumize initialization
def my_init(shape, dtype=None):
    # the initializing weights are taken from a normal dist such that the variance(scale)
    # of the output nodes become equal and have the same contribution throughout the model.
    # This helps to avoid the problem of vanishing or exploding gradients during training.
    # fixing the seed for getting reproducible results
    return tf.random.normal(shape, mean=0, stddev=1 / np.sqrt(shape[0]), dtype=dtype)


# Define the neural network architecture:

# standard keras activations:

def nn_arch(widths: np.ndarray,
            activation: str = 'sigmoid', seed: int = 42):
    """
    This function creates a custom Keras architecture
    param widths: list of widths of layers of the NN
    activation: activation function for all layers
    """
    tf.random.set_seed(seed)  # fixing the random seed

    model = Sequential()
    model.add(Dense(widths[0], input_dim=X_train.shape[1], activation=activation,
                    kernel_initializer=my_init))
    for i in range(len(widths) - 1):
        model.add(Dense(widths[i + 1], activation=activation,
                        kernel_initializer=my_init))
    # because we are doing stuff for regression, our output layer is always a single neuron with linear activation
    model.add(Dense(1, kernel_initializer=my_init))
    return model


def fit_and_predict_using_nn(widths: np.ndarray,
                             activation: str,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             seed: int = 42,
                             use_exponential_decay: bool = False,
                             verbose: int = 1):
    """
    This function compiles, and trains (fits) a neural network, and then predicts
    on the out-of-sample (test) data.
    """
    model = nn_arch(widths=widths,
                    activation=activation, seed=seed)
    # adjusting the learning rate
    initial_learning_rate = 0.00001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.025,
        staircase=True)
    # Compile the model:
    if use_exponential_decay:
        model.compile(optimizer=SGD(learning_rate=lr_schedule), loss='mse')
    else:
        model.compile(optimizer=SGD(learning_rate=initial_learning_rate), loss='mse')

    # Train the model:
    # In Keras, you can compute the validation loss by passing the validation data
    # to the validation_data argument of the fit method. It DOES NOT IMPACT THE TRAIN PROCEDURE.
    # IT IS DONE PURELY FOR INFORMATIONAL PURPOSES
    model.fit(X_train,
              y_train,
              epochs=200,
              batch_size=X_train.shape[0],
              validation_data=(X_test, y_test),
              verbose=verbose)

    # Make predictions on the test data
    y_pred = model.predict(X_test);
    return y_pred


K = 2  # please also try K=5
w = [K * 320, K * 320]  # two hidden layers
y_pred = fit_and_predict_using_nn(widths=w,
                                  activation='relu',  # please also try 'sigmoid'
                                  X_train=X_train,
                                  y_train=y_train,
                                  X_test=X_test,
                                  y_test=y_test,
                                  seed=20,
                                  verbose=0)

# We now train our first neural network. As you will see, we can see both train loss and test (validation) loss as the network is being trained. Watch the huge difference between the two. This is the effect of in-sample overfit. As discovered in [The Virtue of Complexity in Return Prediction](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984925), Timing positions from the NN model are remarkable. They behave similarly to long-only strategies, following the [Camplell and Thomson (2008)](https://dash.harvard.edu/bitstream/1/2622619/2/Campbell_Predicting.pdf) recommendation to impose a non-negativity constraint on expected market returns. But our models learn this behavior as opposed to being handed a constraint. The network is trained using the standard gradient descent [(Backpropagation)](https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76)

plot = True
if plot == True:
    # Let pyplot understand the difference in measures

    min_ = np.min([np.min(y_test), np.min(y_pred)])
    max_ = np.max([np.max(y_test), np.max(y_pred)])
    bins = np.linspace(min_, max_, 100)

    # plot the oos prediction on the test dataset
    plt.hist(y_test, bins, alpha=0.5, label='Test Data')
    plt.hist(y_pred, bins, alpha=0.5, label='Out-of-Sample Prediction')

    plt.legend(loc='upper right')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()

sr_plot(predictions=y_pred, returns=y_test, dates=dates)

# Performance of NN is extremely sensitive to initialization.
# Hence, we now try multiple random initializations and then take an average


fixed_seeds = np.random.randint(low=1, high=1000, size=10)
K = 10
w = [K * 10, K * 10]  # two hidden layers
y_preds = []
names = []

for fixed_seed in fixed_seeds:
    y_preds.append(fit_and_predict_using_nn(widths=w,
                                            activation='sigmoid',
                                            X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                            seed=fixed_seed))
    name = "seed: " + str(fixed_seed)
    names += [name]
names += ["average"]
y_preds.append(np.mean(np.array(y_preds), axis=0))
y_preds = np.squeeze(y_preds)

y_preds.shape

timed_returns = y_preds.T * y_test.reshape(-1, 1)
sharpe_ratios = sharpe_ratio(timed_returns)

plt.hist(np.array(sharpe_ratios), alpha=0.5, label="Sharpe ratios")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
print(sharpe_ratios)

sr_plot(predictions=y_preds.T, returns=y_test, names=names, dates=dates)

# We now perform a NN architecture search. Does the architecture matter for performance? We choose a two-hidden layers architecture, but experiment with the width of each layer. Does combining wide layers with narrow layers, etc., matter?

from itertools import product

w = [16, 32]  # , 64, 128, 256]
depth_of_nn = 2
predictions = []
names = []
for list_of_layer_widths in product(w, repeat=depth_of_nn):
    name = str(list_of_layer_widths)
    names += [name]
    # so now, list_of_layer_widths = list of layer widths. For example, list_of_layer_widths=[16, 32] or list_of_layer_widths=[32, 32], etc
    predictions.append(fit_and_predict_using_nn(widths=np.array(list_of_layer_widths),
                                                activation='relu',
                                                X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                verbose=0))

predictions = np.array(predictions).reshape(X_test.shape[0], len(names))
sr_plot(predictions=predictions, returns=y_test, names=names, dates=dates)

# width of each layer
# list_of_widths_for_the_single_hidden_layer = [100, 700, 1000, 3000, 6000, 10000, 30000, 50000, 70000, 100000, 200000]#, 300000, 400000, 500000]
# list_of_widths_for_the_single_hidden_layer = [5, 10, 11,12,13,14,15,16,17,18,19,20,30,40,50,100, 200, 500, 1000, 5000, 6000, 7000, 10000]
# list_of_widths_for_the_single_hidden_layer=[5*(10**5)]
list_of_widths_for_the_single_hidden_layer = [5, 10, 15, 20]
sharpe_ratios = []
predictions = []
names = []
# first we fix the seed
np.random.seed(0)
# then, we fix the list of seeds that will go into the NN initialization
fixed_seeds = np.random.randint(low=100, size=10)
for width_of_single_hidden_layer in list_of_widths_for_the_single_hidden_layer:
    preds = []
    for i in range(10):
        preds.append(fit_and_predict_using_nn(widths=[width_of_single_hidden_layer],
                                              activation='relu',
                                              X_train=X_train,
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                              seed=fixed_seeds[i],
                                              verbose=0))
    pred = np.mean(np.array(np.squeeze(preds)), axis=0)
    predictions.append(pred)

    # definig the market timing returns
    tmp = np.reshape(pred, (-1, 1)) * y_test.reshape(-1, 1)
    # calculating the Sharpe ratio of timed returns
    sharpe_ratios.append(sharpe_ratio(tmp))

    name = str(width_of_single_hidden_layer)
    names += [name]

plt.plot(np.array(list_of_widths_for_the_single_hidden_layer), np.array(np.squeeze(sharpe_ratios)))
print(sharpe_ratios)
plt.xlabel('width')
plt.ylabel('Sharpe ratio')
plt.title('Sharpe ratio as a function of width size')

plt.show()

sr_plot(predictions=np.transpose(np.squeeze(predictions)), returns=y_test, names=names, dates=dates)
# sr_plot(predictions= predictions, returns=y_test, names = names, dates = dates)