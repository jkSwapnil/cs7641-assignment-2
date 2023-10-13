""" 
Main function for neural network training using Stochastic Gradient Descent
"""

import json
import os
import time

import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from dataset import RiceData

# Global constants
# - Fixed seed to make the result replicable
RANDOM_STATE = 1696346542
RESULT_DIR = "./results/nn_sgd"
PLOT_DIR = "./plots/nn_sgd"
np.random.seed(RANDOM_STATE)


def nn_sgd():
    """ Function to implement neural network training using Stochastic Gradient Descent """
    print("Training neural network using Stochastic Gradient Descent")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # Create the dataset object
    rice_data = RiceData()

    # Train and test the model for different number of iterations
    iterations = []
    training_fitness = []
    testing_fitness = []
    test_results = {}  # Test results for the best weights
    best_cm = None  # Confusion matrix of the best weight
    training_time = 0  # Time taken to train the neural network
    for i in tqdm(range(0, 11)):
        # Set the max iterations
        if i == 0:
            max_iters = 1
            iterations.append(max_iters)
        else:
            max_iters = 10*i
            iterations.append(max_iters)
        # Define the model
        model = MLPClassifier(
            hidden_layer_sizes=[100], activation="logistic", solver="sgd", alpha=0,
            learning_rate_init=0.001, max_iter=max_iters, random_state=RANDOM_STATE
            )
        # Train the neural network
        x_train, y_train = rice_data.get_train()  # Get the train features and labels
        start_time = time.time()  # Start time of training
        model.fit(x_train, y_train)  # Train the model
        end_time = time.time()  # End time of training
        training_time = end_time - start_time
        y_train_pred = model.predict(x_train)
        f1_train = f1_score(y_train, y_train_pred)
        training_fitness.append(f1_train)
        # Generate the testing metrics: % accuracy, f1-score, roc_score, and confusion matrix
        x_test, y_test = rice_data.get_test()  # Get the test features and labels
        y_test_pred = model.predict(x_test)
        y_test_pred_proba = model.predict(x_test)
        percentage_accuracy = accuracy_score(y_test, y_test_pred)*100
        f1_test = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
        testing_fitness.append(f1_test)
        if not test_results or f1_test >= test_results["f1_score"]:
            best_cm = cm
            test_results["accuracy"] = percentage_accuracy
            test_results["f1_score"] = f1_test
            test_results["roc_auc"] = roc_auc
    
    # Save the results
    with open(os.path.join(RESULT_DIR, "result.json"), "w") as f:
        json.dump(test_results, f)
    print("- Test results saved at: ", os.path.join(RESULT_DIR, "result.json"))
    
    # Plot the training/testing fitness vs iterations curve
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(iterations, training_fitness, label="Training Fitness")
    ax.plot(iterations, testing_fitness, label="Testing Fitness")
    ax.set_title("Fitness VS Iterations for Stochastic Gradient Descent")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "training_fitness_vs_iterations.png"))
    plt.close(fig)
    print("- Training curve saved at: ", os.path.join(PLOT_DIR, "training_fitness_vs_iterations.png"))

    # Plot the confusion matrix of the best model
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=rice_data.label_names)
    cm_disp.plot()
    fig = cm_disp.figure_
    fig.set_figwidth(8)
    fig.set_figheight(6)
    ax = cm_disp.ax_
    ax.set_title("Confusion matrix for Stochastic Gradient Descent")
    fig.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close(fig)
    print("- Confusion matrix plot saved at: ", os.path.join(PLOT_DIR, "confusion_matrix.png"))

    return training_time


if __name__ == "__main__":

    import time

    print()
    start_time = time.time()
    training_time = nn_sgd()
    end_time = time.time()
    print(f"Time taken for for the longest training run (100 iters): {training_time:.2f} s")
    print(f"Time taken for all repeated training iterations: {end_time - start_time:.2f} s")
    print()
