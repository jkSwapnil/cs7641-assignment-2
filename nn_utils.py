# Neural network architecture
# [7, 100]

import math

import numpy as np
from sklearn.metrics import f1_score


def sigmoid(x):
    """ Sigmoid function | y = 1/(1 + e^(-x)) 
    Parameters:
        x: (int)
    """
    return 1/(1 + np.exp(-x))


def threshold(x):
    """ Return 1 if x >= 0.5 else 0
    Parameters:
        x: (int)
    """
    return 1 if x >= 0.5 else 0


def feed_forward(state, input_dim, hidden_dim, output_dim, X, predict_proba=False):
    """ Execute feed forward of the 1 hidden layer Neural Network
    Parameters:
        state: network weights as flattened out array (np.ndarray[input_dim * hidden_dim + hidden_dim * output_dim])
        input_dim: size of the input vector (int)
        hidden_dim: size of the hidden vector (int)
        output_dim: size of the output vector (int)
        X: input data (np.ndarray[None, input_dim])
        predict_proba: True to return probabilities instead of class labels (bool) 
    Returns:
        O: np.array[None, 1]
        Output of the feed forward
    """
    # Deflatten the model weights
    w01_size = input_dim * hidden_dim
    w12_size = hidden_dim * output_dim
    assert w01_size + w12_size == len(state), "No enough value to de-flatten the model weights"
    W01 = state[0: w01_size].reshape(input_dim, hidden_dim)
    W12 = state[w01_size: w01_size + w12_size].reshape(hidden_dim, output_dim)

    # Pass the data from network
    A1 = sigmoid(np.matmul(X, W01))
    O = sigmoid(np.matmul(A1, W12))

    if predict_proba:
        return O
    else:
        thresh = np.vectorize(threshold)
        return thresh(O)


def f1_fitness(state, input_dim, hidden_dim, output_dim, data):
    """ Implement F1 score as fitness function for 1 hidden layer neural network 
    Parameters:
        state: network weights as flattened out array (np.ndarray[input_dim * hidden_dim + hidden_dim * output_dim])
        input_dim: size of the input vector (int)
        hidden_dim: size of the hidden vector (int)
        output_dim: size of the output vector (int)
        data: input data (Data)
    Returns:
        F1 score
    """
    # Get the X and y_true 
    X, y_true = data.get_train()
    
    # Generate output from the neural netork
    y_pred = np.squeeze(feed_forward(state, input_dim, hidden_dim, output_dim, X))

    return f1_score(y_true, y_pred)


if __name__ == "__main__":

    import numpy as np
    import mlrose_hiive as mlrose

    from dataset import RiceData

    # Test the feed forward
    print()
    print("Testing the feed forward function")
    X = np.array([[1,1],[1,0]])
    expected = np.array([[0.99290395], [0.98119724]])
    out = feed_forward(np.array([1,2,3,4,3,2]), input_dim=2, hidden_dim=2, output_dim=1, X=X)
    if np.all(expected - out < 0.001):
        print("- pass")
    else:
        print("- fail")

    # Test the fitness function
    print("\nTesting the fitness function")
    rice_data = RiceData()
    input_dim = 7 
    hidden_dim = 100
    output_dim = 1
    state_init = np.random.normal(loc=0, scale=1, size=input_dim * hidden_dim + hidden_dim * output_dim)
    train_kwargs = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "data": rice_data}
    train_f1_score = f1_fitness(state_init, **train_kwargs)
    test_kwargs = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "data": rice_data}
    test_f1_score = f1_fitness(state_init, **test_kwargs)
    print(f"- Train F1 score: {train_f1_score}")
    print(f"- Test F1 score: {test_f1_score}")

    # Test RO using Randomized Hill Climb
    print("\nTesting the training using Randomized Hill Climbing")
    rice_data = RiceData()
    input_dim = 7 
    hidden_dim = 100
    output_dim = 1
    kwargs = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "data": rice_data}
    fitness = mlrose.CustomFitness(fitness_fn=f1_fitness, **kwargs)  # Create the fitness function object
    n = input_dim * hidden_dim + hidden_dim * output_dim  # Set the problem size
    state_init = np.random.normal(loc=0, scale=10, size=n)  # Initialize the intial state
    problem = mlrose.ContinuousOpt(  # Create optmization problem object
        length=n, fitness_fn=fitness, maximize=True, min_val=-100, max_val=100, step=1
        )
    rhc_best_state, rhc_best_fitness, _ = mlrose.random_hill_climb(problem=problem, max_attempts=100, max_iters=10000, curve=True)
    print(f"- best fitness: {rhc_best_fitness}")
    print()
