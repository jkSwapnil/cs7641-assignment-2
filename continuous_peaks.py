"""
This module implements the optimization for ContinuousPeaks problems using the following RO algorithms
    - RHC: Randomized Hill Climb
    - SA: Simulated Annealing
    - GA: Genetic Algorithm
    - MIMIC
"""

import json
import os
import random
import statistics
import time

import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import numpy as np
from tqdm import tqdm


# Global constants
# - Fixed seed to make the result replicable
# - Directory location to save the results as JSON
# - Directory location for saving the plots
RANDOM_STATE = 1696346542
RESULT_DIR = "./results/continuous_peaks"
PLOT_DIR = "./plots/continuous_peaks"


def fitness_vs_iteration_analysis():
    """ Plot fitness values vs iterations for the ContinuousPeaks problem.

    Imp. points:
        - We use the binary string state for this problem.
        - For plotting fitness vs iteration, length=100 is used.
        - The number of iterations are capped at 500
    """
    print("Fitness VS Iterations")
    print("- - - - - - - - - - -")
    # Define commone settings related to the optimation algorithms
    n = 50  # Size of the problem (length of the input state)
    max_attempts = 100   # Max number of attempts to find better point in neighbourhood
    max_iters = 500      # Maximum number of iterations for optimization
    schedule = mlrose.ExpDecay()  # Temperature decay for the Simulated Annealing

    # Define fitness function and optimization problem
    fitness = mlrose.ContinuousPeaks(t_pct=0.1)  # Create the fitness function object
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)  # Create optmization problem object

    # Define the result dict to save the result
    result = {}

    # Optimizing using Randomized Hill Climbing
    rhc_best_state, rhc_best_fitness, rhc_fitness_vs_iter_curve = mlrose.random_hill_climb(
        problem=problem, max_attempts=max_attempts, max_iters=max_iters, curve=True, random_state=RANDOM_STATE
        )
    result["Random Hill Climb"] = {"Best fitness": rhc_best_fitness, "Iterations": rhc_fitness_vs_iter_curve.shape[0]}
    print("- Random Hill Climb: ")
    print(f"\t- Best fitness: {rhc_best_fitness}")
    print(f"\t- Iterations: {rhc_fitness_vs_iter_curve.shape[0]}")
    
    # Optimizing using Simulated Annealing
    sa_best_state, sa_best_fitness, sa_fitness_vs_iter_curve = mlrose.simulated_annealing(
        problem=problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, curve=True, random_state=RANDOM_STATE)
    result["Simulated Annealing"] = {"Best fitness": sa_best_fitness, "Iterations": sa_fitness_vs_iter_curve.shape[0]}
    print("- Simulated Annealing: ")
    print(f"\t- Best fitness: {sa_best_fitness}")
    print(f"\t- Iterations: {sa_fitness_vs_iter_curve.shape[0]}")

    # Optimizing using Genetic Algorithm
    ga_best_state, ga_best_fitness, ga_fitness_vs_iter_curve = mlrose.genetic_alg(
        problem=problem, pop_size=200, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, curve=True, 
        random_state=RANDOM_STATE
        )
    result["Genetic Algorithm"] = {"Best fitness": ga_best_fitness, "Iterations": ga_fitness_vs_iter_curve.shape[0]}
    print("- Genetic Algorithm: ")
    print(f"\t- Best fitness: {ga_best_fitness}")
    print(f"\t- Iterations: {ga_fitness_vs_iter_curve.shape[0]}")

    # Optimizing using MIMIC
    mimic_best_state, mimic_best_fitness, mimic_fitness_vs_iter_curve = mlrose.mimic(
        problem=problem, pop_size=8000, keep_pct=0.2, max_attempts=max_attempts/5, max_iters=max_iters, curve=True, 
        random_state=RANDOM_STATE
        )
    result["MIMIC"] = {"Best fitness": mimic_best_fitness, "Iterations": mimic_fitness_vs_iter_curve.shape[0]}
    print("- MIMIC: ")
    print(f"\t- Best fitness: {mimic_best_fitness}")
    print(f"\t- Iterations: {mimic_fitness_vs_iter_curve.shape[0]}")

    # Save the result dict as JSON object
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    with open(os.path.join(RESULT_DIR, "fitness_vs_iterations.json"), "w") as f:
        json.dump(result, f)
    print("- Results saved at: " + os.path.join(RESULT_DIR, "fitness_vs_iterations.json"))

    # Plot the fitness vs iters curve for the algorithms
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(1, rhc_fitness_vs_iter_curve.shape[0]+1), rhc_fitness_vs_iter_curve[:,0], label="Random Hill Climbing")
    ax.plot(range(1, sa_fitness_vs_iter_curve.shape[0]+1), sa_fitness_vs_iter_curve[:,0], label="Simulated Annealing")
    ax.plot(range(1, ga_fitness_vs_iter_curve.shape[0]+1), ga_fitness_vs_iter_curve[:,0], label="Genetic Algorithm")
    ax.plot(range(1, mimic_fitness_vs_iter_curve.shape[0]+1), mimic_fitness_vs_iter_curve[:,0], label="MIMIC")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness vs Iterations")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "fitness_vs_iterations.png"))
    plt.close(fig)
    print("- Plot saved at: " + os.path.join(PLOT_DIR, "fitness_vs_iterations.png"))


def full_sa_fitness_vs_iteration_analysis():
    """ Plot fitness values vs iterations for the ContinuousPeaks problem using full Simulated Annealing.

    Imp. points:
        - We use the binary string state for this problem.
        - Simulated annealing is run longer to converge 
        - For plotting fitness vs iteration, length=100 is used.
        - The number of iterations are capped at 5000
    """
    print("Fitness VS Iterations (Full Simulated Annealing)")
    print("- - - - - - - - - - - - - - - - - - - - - - - - ")
    # Define commone settings related to the optimation algorithms
    n = 50  # Size of the problem (length of the input state)
    max_attempts = 100   # Max number of attempts to find better point in neighbourhood
    max_iters = 5000     # Maximum number of iterations for optimization
    schedule = mlrose.ExpDecay()  # Temperature decay for the Simulated Annealing

    # Define the result dict to save the result
    result = {}

    # Define fitness function and optimization problem
    fitness = mlrose.ContinuousPeaks(t_pct=0.1)  # Create the fitness function object
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)  # Create optmization problem object
    
    # Optimizing using Simulated Annealing
    _, sa_best_fitness, sa_fitness_vs_iter_curve = mlrose.simulated_annealing(
        problem=problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, curve=True, random_state=RANDOM_STATE)
    result["Full Simulated Annealing"] = {"Best fitness": sa_best_fitness, "Iterations": sa_fitness_vs_iter_curve.shape[0]}
    print(f"\t- Best fitness: {sa_best_fitness}")
    print(f"\t- Iterations: {sa_fitness_vs_iter_curve.shape[0]}")

    # Save the results as JSON
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    with open(os.path.join(RESULT_DIR, "full_sa_fitness_vs_iterations.json"), "w") as f:
        json.dump(result, f)
    print("- Results saved at: " + os.path.join(RESULT_DIR, "full_sa_fitness_vs_iterations.json"))

    # Plot the fitness vs iters curve for the algorithms
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(1, sa_fitness_vs_iter_curve.shape[0]+1), sa_fitness_vs_iter_curve[:,0], label="Simulated Annealing")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness vs Iterations")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "full_sa_fitness_vs_iterations.png"))
    plt.close(fig)
    print("- Plot saved at: " + os.path.join(PLOT_DIR, "full_sa_fitness_vs_iterations.png"))


def scalability_analysis():
    """ Perform scalability analysis of ContinuousPeaks problem.

    Imp points:
        - Increase the problem size ContinuousPeaks in the step of 10
        - Plot fitness vs problem size
        - Plot fevals/iterations vs problem size 
        - Wall clock time vs problem size
    """
    print("Scalability Test")
    print("- - - - - - - - ")
    print("- Problem size increased from 10 to 110 in steps of 10")
    # Define the lists to save the statistics
    size = []  # Sizes of the problem
    rhc_best_fitness_avg = []  # Average of the best-fitness from all the trails for each problem size (Random Hill Climbing)
    rhc_best_fitness_std = []  # Std. deviation of the best-fitness from all the trails for each problem size (Random Hill Climbing)
    sa_best_fitness_avg = []  # Average of the best-fitness from all the trails for each problem size (Simulated Annealing)
    sa_best_fitness_std = []  # Std. deviation of the best-fitness from all the trails for each problem size (Simulated Annealing)
    ga_best_fitness_avg = []  # Average of the best-fitness from all the trails for each problem size (Genetic Algorithm)
    ga_best_fitness_std = []  # Std. deviation of the best-fitness from all the trails for each problem size (Genetic Algorithm)
    mimic_best_fitness_avg = []  # Average of the best-fitness from all the trails for each problem size (MIMIC)
    mimic_best_fitness_std = []  # Std. deviation of the best-fitness from all the trails for each problem size (MIMIC)
    rhc_feval_per_iteration_avg = []  # Averge of feval/iteration from all the trails for each problem size (Random Hill Climbing)
    rhc_feval_per_iteration_std = []  # Stdev of feval/iteration from all the trails for each problem size (Random Hill Climbing)
    sa_feval_per_iteration_avg = []  # Averge of feval/iteration from all the trails for each problem size (Simulated Annealing)
    sa_feval_per_iteration_std = []  # Stdev of feval/iteration from all the trails for each problem size (Simulated Annealing)
    ga_feval_per_iteration_avg = []  # Averge of feval/iteration from all the trails for each problem size (Genetic Algorithm)
    ga_feval_per_iteration_std = []  # Stdev of feval/iteration from all the trails for each problem size (Genetic Algorithm)
    mimic_feval_per_iteration_avg = []  # Averge of feval/iteration from all the trails for each problem size (MIMIC)
    mimic_feval_per_iteration_std = []  # Std. deviation of feval/iteration from all the trails for each problem size (MIMIC)
    rhc_wall_clock_avg = []  # Avg. wall clock time from all the trails for each problem size (Random Hill Climbing)
    rhc_wall_clock_std = []  # Std. deviation of wall clock time from all the trails for each problem size (Random Hill Climbing)
    sa_wall_clock_avg = []  # Avg. wall clock time from all the trails for each problem size (Simulated Annealing)
    sa_wall_clock_std = []  # Std. deviation of wall clock time from all the trails for each problem size (Simulated Annealing)
    ga_wall_clock_avg = []  # Avg. wall clock time from all the trails for each problem size (Genetic Algorithm)
    ga_wall_clock_std = []  # Std. deviation of wall clock time from all the trails for each problem size (Genetic Algorithm)
    mimic_wall_clock_avg = []  # Avg. wall clock time from all the trails for each problem size (MIMIC)
    mimic_wall_clock_std = []  # Std. deviation of wall clock time from all the trails for each problem size (MIMIC)

    # Loop for increaring the function size
    for n in tqdm(range(5, 56, 5)):
        # Define fitness function and optimization problem
        fitness = mlrose.ContinuousPeaks(t_pct=0.1)
        problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
        schedule = mlrose.ExpDecay()

        # Hyperparameters for the optimization problem
        max_attempts = 100
        max_iters = 500

        # Loop for multiple trails of same problem size
        trail_rhc_best_fitness = []
        trail_rhc_feval_per_iteration = []
        trail_rhc_wall_clock_time = []
        trail_sa_best_fitness = []
        trail_sa_feval_per_iteration = []
        trail_sa_wall_clock_time = []
        trail_ga_best_fitness = []
        trail_ga_feval_per_iteration = []
        trail_ga_wall_clock_time = []
        trail_mimic_best_fitness = []
        trail_mimic_feval_per_iteration = []
        trail_mimic_wall_clock_time = []
        for trail in range(5):
            # Optimizing using Randomized Hill Climbing
            rhc_begin_time = time.time()
            _, rhc_best_fitness, rhc_fitness_vs_iter_curve = mlrose.random_hill_climb(
                problem=problem, max_attempts=max_attempts, max_iters=max_iters, curve=True, random_state=RANDOM_STATE + trail
            )
            rhc_end_time = time.time()
            rhc_feval = rhc_fitness_vs_iter_curve[-1, 1]
            rhc_iterations = rhc_fitness_vs_iter_curve.shape[0]
            rhc_wall_clock_time = rhc_end_time - rhc_begin_time
            trail_rhc_best_fitness.append(rhc_best_fitness/n)
            trail_rhc_feval_per_iteration.append(rhc_feval/rhc_iterations)
            trail_rhc_wall_clock_time.append(rhc_wall_clock_time)
            # Optimizing using Simulated Annealing
            sa_begin_time = time.time()
            _, sa_best_fitness, sa_fitness_vs_iter_curve = mlrose.simulated_annealing(
                problem=problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters*10, curve=True, 
                random_state=RANDOM_STATE + trail
            )
            sa_end_time = time.time()
            sa_feval = sa_fitness_vs_iter_curve[-1, 1]
            sa_iterations = sa_fitness_vs_iter_curve.shape[0]
            sa_wall_clock_time = sa_end_time - sa_begin_time
            trail_sa_best_fitness.append(sa_best_fitness/n)
            trail_sa_feval_per_iteration.append(sa_feval/sa_iterations)
            trail_sa_wall_clock_time.append(sa_wall_clock_time)
            # Optimizing using Genetic Algorithm
            ga_start_time = time.time()
            _, ga_best_fitness, ga_fitness_vs_iter_curve = mlrose.genetic_alg(
                problem=problem, pop_size=200, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, curve=True, 
                random_state=RANDOM_STATE + trail
            )
            ga_end_time = time.time()
            ga_feval = ga_fitness_vs_iter_curve[-1, 1]
            ga_iterations = ga_fitness_vs_iter_curve.shape[0]
            ga_wall_clock_time = ga_end_time - ga_start_time
            trail_ga_best_fitness.append(ga_best_fitness/n)
            trail_ga_feval_per_iteration.append(ga_feval/ga_iterations)
            trail_ga_wall_clock_time.append(ga_wall_clock_time)
            # Optimizing using MIMIC
            mimic_start_time = time.time()
            _, mimic_best_fitness, mimic_fitness_vs_iter_curve = mlrose.mimic(
                problem=problem, pop_size=200, keep_pct=0.2, max_attempts=max_attempts/10, max_iters=max_iters, curve=True, 
                random_state=RANDOM_STATE + trail
            )
            mimic_end_time = time.time()
            mimic_feval = mimic_fitness_vs_iter_curve[-1, 1]
            mimic_iterations = mimic_fitness_vs_iter_curve.shape[0]
            mimic_wall_clock_time = mimic_end_time - mimic_start_time
            trail_mimic_best_fitness.append(mimic_best_fitness/n)
            trail_mimic_feval_per_iteration.append(mimic_feval/mimic_iterations)
            trail_mimic_wall_clock_time.append(mimic_wall_clock_time)

        # Added the value for the problem size
        size.append(n)
        rhc_best_fitness_avg.append(statistics.mean(trail_rhc_best_fitness))
        rhc_best_fitness_std.append(statistics.stdev(trail_rhc_best_fitness))
        sa_best_fitness_avg.append(statistics.mean(trail_sa_best_fitness))
        sa_best_fitness_std.append(statistics.stdev(trail_sa_best_fitness))
        ga_best_fitness_avg.append(statistics.mean(trail_ga_best_fitness))
        ga_best_fitness_std.append(statistics.stdev(trail_ga_best_fitness))
        mimic_best_fitness_avg.append(statistics.mean(trail_mimic_best_fitness))
        mimic_best_fitness_std.append(statistics.stdev(trail_mimic_best_fitness))
        rhc_feval_per_iteration_avg.append(statistics.mean(trail_rhc_feval_per_iteration))
        rhc_feval_per_iteration_std.append(statistics.stdev(trail_rhc_feval_per_iteration))
        sa_feval_per_iteration_avg.append(statistics.mean(trail_sa_feval_per_iteration))
        sa_feval_per_iteration_std.append(statistics.stdev(trail_sa_feval_per_iteration))
        ga_feval_per_iteration_avg.append(statistics.mean(trail_ga_feval_per_iteration))
        ga_feval_per_iteration_std.append(statistics.stdev(trail_ga_feval_per_iteration))
        mimic_feval_per_iteration_avg.append(statistics.mean(trail_mimic_feval_per_iteration))
        mimic_feval_per_iteration_std.append(statistics.stdev(trail_mimic_feval_per_iteration))
        rhc_wall_clock_avg.append(statistics.mean(trail_rhc_wall_clock_time))
        rhc_wall_clock_std.append(statistics.stdev(trail_rhc_wall_clock_time))
        sa_wall_clock_avg.append(statistics.mean(trail_sa_wall_clock_time))
        sa_wall_clock_std.append(statistics.stdev(trail_sa_wall_clock_time))
        ga_wall_clock_avg.append(statistics.mean(trail_ga_wall_clock_time))
        ga_wall_clock_std.append(statistics.stdev(trail_ga_wall_clock_time))
        mimic_wall_clock_avg.append(statistics.mean(trail_mimic_wall_clock_time))
        mimic_wall_clock_std.append(statistics.stdev(trail_mimic_wall_clock_time))

    # Convert to numpy array
    size = np.array(size)
    rhc_best_fitness_avg = np.array(rhc_best_fitness_avg)
    rhc_best_fitness_std = np.array(rhc_best_fitness_std)
    sa_best_fitness_avg = np.array(sa_best_fitness_avg)
    sa_best_fitness_std = np.array(sa_best_fitness_std)
    ga_best_fitness_avg = np.array(ga_best_fitness_avg)
    ga_best_fitness_std = np.array(ga_best_fitness_std)
    mimic_best_fitness_avg = np.array(mimic_best_fitness_avg)
    mimic_best_fitness_std = np.array(mimic_best_fitness_std)
    rhc_feval_per_iteration_avg = np.array(rhc_feval_per_iteration_avg)
    rhc_feval_per_iteration_std = np.array(rhc_feval_per_iteration_std)
    sa_feval_per_iteration_avg = np.array(sa_feval_per_iteration_avg)
    sa_feval_per_iteration_std = np.array(sa_feval_per_iteration_std)
    ga_feval_per_iteration_avg = np.array(ga_feval_per_iteration_avg)
    ga_feval_per_iteration_std = np.array(ga_feval_per_iteration_std)
    mimic_feval_per_iteration_avg = np.array(mimic_feval_per_iteration_avg)
    mimic_feval_per_iteration_std = np.array(mimic_feval_per_iteration_std)
    rhc_wall_clock_avg = np.array(rhc_wall_clock_avg)
    rhc_wall_clock_std = np.array(rhc_wall_clock_std)
    sa_wall_clock_avg = np.array(sa_wall_clock_avg)
    sa_wall_clock_std = np.array(sa_wall_clock_std)
    ga_wall_clock_avg = np.array(ga_wall_clock_avg)
    ga_wall_clock_std = np.array(ga_wall_clock_std)
    mimic_wall_clock_avg = np.array(mimic_wall_clock_avg)
    mimic_wall_clock_std = np.array(mimic_wall_clock_std)

    # Create the appropriate plots
    # Generate plots for best fitness VS problem size
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(size, rhc_best_fitness_avg, label="Randomized Hill Climb")
    ax.fill_between(size, rhc_best_fitness_avg - rhc_best_fitness_std, rhc_best_fitness_avg + rhc_best_fitness_std, alpha=0.5)
    ax.plot(size, sa_best_fitness_avg, label="Simulated Annealing")
    ax.fill_between(size, sa_best_fitness_avg - sa_best_fitness_std, sa_best_fitness_avg + sa_best_fitness_std, alpha=0.5)
    ax.plot(size, ga_best_fitness_avg, label="Genetic Algorithm")
    ax.fill_between(size, ga_best_fitness_avg - ga_best_fitness_std, ga_best_fitness_avg + ga_best_fitness_std, alpha=0.5)
    ax.plot(size, mimic_best_fitness_avg, label="MIMIC")
    ax.fill_between(size, mimic_best_fitness_avg - mimic_best_fitness_std, mimic_best_fitness_avg + mimic_best_fitness_std, alpha=0.5)
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Scaled fitness (Fitness/n)")
    ax.set_title("Scaled fitness VS Problem size")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "fitness_vs_size.png"))
    plt.close(fig)
    print("- Plot saved at: " + os.path.join(PLOT_DIR, "fitness_vs_size.png"))

    # Generate plots for feval/iteration VS problem size
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(size, rhc_feval_per_iteration_avg, label="Randomized Hill Climb")
    ax.fill_between(
        size, 
        rhc_feval_per_iteration_avg - rhc_feval_per_iteration_std, 
        rhc_feval_per_iteration_avg + rhc_feval_per_iteration_std, 
        alpha=0.5
        )
    ax.plot(size, sa_feval_per_iteration_avg, label="Simulated Annealing")
    ax.fill_between(
        size, 
        sa_feval_per_iteration_avg - sa_feval_per_iteration_std, 
        sa_feval_per_iteration_avg + sa_feval_per_iteration_std, 
        alpha=0.5
        )
    ax.plot(size, ga_feval_per_iteration_avg, label="Genetic Algorithm")
    ax.fill_between(
        size, 
        ga_feval_per_iteration_avg - ga_feval_per_iteration_std, 
        ga_feval_per_iteration_avg + ga_feval_per_iteration_std, 
        alpha=0.5
        )
    ax.plot(size, mimic_feval_per_iteration_avg, label="MIMIC")
    ax.fill_between(
        size, 
        mimic_feval_per_iteration_avg - mimic_feval_per_iteration_std, 
        mimic_feval_per_iteration_avg + mimic_feval_per_iteration_std, 
        alpha=0.5
        )
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Feval/Iteration")
    ax.set_title("Feval/Iteration VS Problem size")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "feval_per_iteration_vs_size.png"))
    plt.close(fig)
    print("- Plot saved at: " + os.path.join(PLOT_DIR, "feval_per_iteration_vs_size.png"))

    # Generate plots for wall_clock_time VS problem size
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(size, rhc_wall_clock_avg, label="Randomized Hill Climb")
    ax.fill_between(size, rhc_wall_clock_avg - rhc_wall_clock_std, rhc_wall_clock_avg + rhc_wall_clock_std, alpha=0.5)
    ax.plot(size, sa_wall_clock_avg, label="Simulated Annealing")
    ax.fill_between(size, sa_wall_clock_avg - sa_wall_clock_std, sa_wall_clock_avg + sa_wall_clock_std, alpha=0.5)
    ax.plot(size, ga_wall_clock_avg, label="Genetic Algorithm")
    ax.fill_between(size, ga_wall_clock_avg - ga_wall_clock_std, ga_wall_clock_avg + ga_wall_clock_std, alpha=0.5)
    ax.plot(size, mimic_wall_clock_avg, label="MIMIC")
    ax.fill_between(size, mimic_wall_clock_avg - mimic_wall_clock_std, mimic_wall_clock_avg + mimic_wall_clock_std, alpha=0.5)
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Wall clock time (s)")
    ax.set_title("Wall clock time VS Problem size")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "wall_clock_time_vs_size.png"))
    plt.close(fig)
    print("- Plot saved at: " + os.path.join(PLOT_DIR, "wall_clock_time_vs_size.png"))


if __name__ == "__main__":

    print()
    fitness_vs_iteration_analysis()
    print()
    full_sa_fitness_vs_iteration_analysis()
    print()
    scalability_analysis()
    print()
