import random
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import concurrent.futures
from scipy.optimize import curve_fit
import multiprocessing

output_folder = "results"
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

def format_time(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} hours"
    elif seconds >= 60:
        return f"{seconds / 60:.2f} minutes"
    else:
        return f"{seconds:.2f} seconds"

# Define experiment and run values to test
fixed_run_value = 200  # Fixed run value
experiment_range = (1, 2500)  # Define range for experiments
number_range = (1, 10)
num_repeats = 1  # Number of times each combination runs
test_all_numbers = False  # Enable testing all numbers between the range
test_all_experiments = True  # Enable testing all experiments between the range

# Function to perform a single experiment and return randomness score
def run_experiment(experiments, runs, num_range):
    start_time = time.time()
    
    winners = {str(i): 0 for i in range(num_range[0], num_range[1] + 1)}
    winners['draw'] = 0

    for _ in range(experiments):
        counts = {i: 0 for i in range(num_range[0], num_range[1] + 1)}
        for _ in range(runs):
            num = random.randint(num_range[0], num_range[1])
            counts[num] += 1

        max_count = max(counts.values())
        winner_nums = [str(k) for k, v in counts.items() if v == max_count]

        if len(winner_nums) == 1:
            winners[winner_nums[0]] += 1
        else:
            winners['draw'] += 1

    # Calculate randomness score (lower standard deviation means more random spread)
    counts_values = np.array(list(winners.values()))
    randomness_score = np.std(counts_values)

    elapsed_time = time.time() - start_time
    return (experiments, runs, randomness_score, elapsed_time)

if __name__ == "__main__":
    total_start_time = time.time()
    manager = multiprocessing.Manager()
    randomness_scores = manager.dict()
    execution_times = manager.dict()
    
    number_ranges_to_test = [number_range]
    if test_all_numbers:
        number_ranges_to_test = [(i, j) for i in range(number_range[0], number_range[1]) for j in range(i + 1, number_range[1] + 1)]
    
    experiment_values_to_test = [i for i in range(experiment_range[0], experiment_range[1] + 1)]
    
    def process_experiment(experiments, num_range):
        exp, runs, score, elapsed = run_experiment(experiments, fixed_run_value, num_range)
        key = f"Experiments: {experiments}, Range: {num_range}"
        with manager.Lock():
            if key not in randomness_scores:
                randomness_scores[key] = []
            randomness_scores[key].append(score)
            execution_times[experiments] = elapsed
        print(f"Completed {key} - Time taken: {format_time(elapsed)}", flush=True)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for num_range in number_ranges_to_test:
            for _ in range(num_repeats):
                for experiments in experiment_values_to_test:
                    futures.append(executor.submit(process_experiment, experiments, num_range))
        concurrent.futures.wait(futures)

    total_end_time = time.time()

    # Compute average randomness scores
    experiment_values = np.array(sorted(set(experiment_values_to_test)))
    average_randomness_scores = np.array([
        np.mean(randomness_scores.get(f"Experiments: {experiments}, Range: {number_range}", [0]))
        if randomness_scores.get(f"Experiments: {experiments}, Range: {number_range}") else 0
        for experiments in experiment_values
    ])

    # Define a polynomial function for curve fitting
    def poly_fit(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit the curve
    if np.any(average_randomness_scores):  # Ensure no all-zero data
        params, _ = curve_fit(poly_fit, experiment_values, average_randomness_scores)
        fitted_curve = poly_fit(experiment_values, *params)
    else:
        fitted_curve = np.zeros_like(experiment_values)

    # Plot the randomness score with a curved line of best fit
    plt.figure()
    plt.scatter(experiment_values, average_randomness_scores, color='blue', label='Data Points')
    plt.plot(experiment_values, fitted_curve, color='red', linestyle='--', label='Best Fit Curve')
    plt.xlabel('Number of Experiments')
    plt.ylabel('Average Randomness Score (Standard Deviation)')
    plt.title(f'Randomness Score vs Experiments (Runs: {fixed_run_value})')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'randomness_best_fit_curve.png'))
    plt.show()
    
    # Plot execution time vs experiments
    execution_time_values = np.array([execution_times.get(exp, 0) for exp in experiment_values])
    plt.figure()
    plt.plot(experiment_values, execution_time_values, color='green', marker='o', linestyle='-', label='Execution Time')
    plt.xlabel('Number of Experiments')
    plt.ylabel('Execution Time (Seconds)')
    plt.title('Execution Time vs Number of Experiments')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'execution_time_curve.png'))
    plt.show()
    
    print(f"Total execution time: {format_time(total_end_time - total_start_time)}")
