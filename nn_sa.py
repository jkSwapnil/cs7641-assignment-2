""" 
Main function for neural network training using Simulated Annealing
"""

import json
import os

import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from dataset import RiceData
from nn_utils import f1_fitness, feed_forward

# Global constants
# - Fixed seed to make the result replicable
RANDOM_STATE = 1696346542
RESULT_DIR = "./results/nn_sa"
PLOT_DIR = "./plots/nn_sa"
np.random.seed(RANDOM_STATE)


def nn_sa():
    """ Function to implement neural network training using Simulated Annealing """
    print("Training neural network using Simulated Annealing")
    print("- - - - - - - - - - - - - - - - - - - - - - - - ")

    # Create the fitness function object and problem object
    rice_data = RiceData()
    input_dim = 7
    hidden_dim = 100
    output_dim = 1
    train_kwargs = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "data": rice_data}
    fitness = mlrose.CustomFitness(fitness_fn=f1_fitness, **train_kwargs)  # Create the fitness function object
    n = input_dim * hidden_dim + hidden_dim * output_dim  # Set the problem size
    state_init = np.random.normal(loc=0, scale=10, size=n)  # Initialize the intial state
    problem = mlrose.ContinuousOpt(  # Create optmization problem object
        length=n, fitness_fn=fitness, maximize=True, min_val=-100, max_val=100, step=1
        )

    # Schedule of temperature decay
    schedule = mlrose.ExpDecay()

    # Train and test the model for different number of iterations
    iterations = []
    training_fitness = []
    testing_fitness = []
    test_results = {}  # Test results for the best weights
    best_cm = None  # Confusion matrix of the best weight
    cur_state = np.random.normal(loc=0, scale=10, size=n)  # Initialize the state
    for i in tqdm(range(0, 81)):
        # Set the max iterations
        if i == 0:
            max_iters = 0
            iterations.append(max_iters)
        else:
            max_iters = 100
            iterations.append(max_iters*i)
        # Train the neural network
        sa_best_state, sa_best_fitness, _ = mlrose.simulated_annealing(
            problem=problem, schedule=schedule, max_attempts=1000, max_iters=max_iters, 
            init_state=cur_state, curve=True, random_state=RANDOM_STATE
            )
        training_fitness.append(sa_best_fitness)
        # Generate the testing metrics: % accuracy, f1-score, roc_score, and confusion matrix
        x_test, y_test = rice_data.get_test()  # Get the test features and labels
        y_pred = feed_forward(  # Generate label predictions on test data
            state=sa_best_state, input_dim=input_dim, hidden_dim=hidden_dim, 
            output_dim=output_dim, X=x_test, predict_proba=False
            )
        y_pred_proba = feed_forward(  # Generate probability predictions on test data
            state=sa_best_state, input_dim=input_dim, hidden_dim=hidden_dim, 
            output_dim=output_dim, X=x_test, predict_proba=True
            )
        percentage_accuracy = accuracy_score(y_test, y_pred)*100
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        testing_fitness.append(f1)
        if not test_results or f1 >= test_results["f1_score"]:
            best_cm = cm
            test_results["accuracy"] = percentage_accuracy
            test_results["f1_score"] = f1
            test_results["roc_auc"] = roc_auc
        cur_state = sa_best_state
    
    # Save the results
    with open(os.path.join(RESULT_DIR, "result.json"), "w") as f:
        json.dump(test_results, f)
    print("- Test results saved at: ", os.path.join(RESULT_DIR, "result.json"))
    
    # Plot the training/testing fitness vs iterations curve
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(iterations, training_fitness, label="Training Fitness")
    ax.plot(iterations, testing_fitness, label="Testing Fitness")
    ax.set_title("Fitness VS Iterations for Simulated Annealing")
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
    ax.set_title("Confusion matrix for Simulated Annealing")
    fig.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close(fig)
    print("- Confusion matrix plot saved at: ", os.path.join(PLOT_DIR, "confusion_matrix.png"))


if __name__ == "__main__":

    import time

    print()
    start_time = time.time()
    nn_sa()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} s")
    print()
    