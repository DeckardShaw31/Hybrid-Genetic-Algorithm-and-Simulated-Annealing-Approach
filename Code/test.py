from penalty import fitness_function
from GA import generate_chromosome
from GA import check_validation
from GA import repair_chromosome
from GA import tournament_selection
from GA import one_point_crossover
from GA import swap_mutation
from GA import replacement
from penalty import chromosome_to_schedule
from ED_DES import run_emergency_dept

import numpy as np
import pandas as pd
import random
import simpy
import scipy.stats

num_generation = 15
population_size = 15
tournament_size = 3
elitism_size = 3
crossover_probability = 1
mutation_probability = 0.2
num_fulltime_shifts = 3
num_parttime_shifts = 6
num_consultant_rooms = 6
len_parttime_shift = 3
start_parttime_shift = [8, 9, 10, 14, 15, 19]
alpha = 1.5
beta = 0.8
gamma = 1
chromosome = [3, 4, 2, 0, 1, 0, 0, 0, 0]


'''population = []
fitness_scores = []

# generate initial population and get fitness scores
for i in range(population_size):
    for attempt in range(100):  # Limit attempts to avoid infinite loop
        chromosome = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)
        result, invalid_constraint = check_validation(chromosome, num_consultant_rooms, len_parttime_shift,
                                                      start_parttime_shift)
        if result == True:
            population.append(chromosome)
            score = fitness_function(chromosome, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)
            fitness_scores.append(score)
            break

fitness_scores = np.array(fitness_scores)
population = np.array(population)

population = population[fitness_scores.argsort()]
fitness_scores = fitness_scores[fitness_scores.argsort()]
best_chromosome = []
best_score = []

# selection
parent1, parent2 = tournament_selection(population, fitness_scores, tournament_size)

# crossover
if random.random() <= crossover_probability:
    child1, child2 = one_point_crossover(parent1, parent2)
else:
    child1 = parent1
    child2 = parent2

# repair chromosome
child1 = repair_chromosome(child1, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
child2 = repair_chromosome(child2, num_consultant_rooms, len_parttime_shift, start_parttime_shift)

# mutation
if random.random() <= mutation_probability:
    child1 = swap_mutation(child1)
if random.random() <= mutation_probability:
    child2 = swap_mutation(child2)

# repair chromosome
child1 = repair_chromosome(child1, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
child2 = repair_chromosome(child2, num_consultant_rooms, len_parttime_shift, start_parttime_shift)

# penalty evaluation
score1 = fitness_function(child1, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)
score2 = fitness_function(child2, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)

# replacement
population, fitness_scores = replacement(population, fitness_scores, child1, child2, score1, score2)

# store value
best_chromosome.append(population[0])
best_score.append(fitness_scores[0])'''

file_path = "Data1.xlsx"
df_service_times = pd.read_excel(file_path, sheet_name="Service times")
service_times = df_service_times.iloc[:, 0].tolist()

df_arrival_rates = pd.read_excel(file_path, sheet_name=f"Arrival rates Day 0")
arrival_rates = df_arrival_rates.iloc[:, 0].tolist()

# # run simulation for stochastics input
# all_run_total_wait_times = []
#
# for i in range(100):
#     env = simpy.Environment()
#     stats = {}
#     patient_log = []
#     doctor_schedule = chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift)
#
#     env.process(
#         run_emergency_dept(env, stats=stats, num_triage_nurses=1, doctor_schedule=doctor_schedule, num_lab_tests=1,
#                            num_ultrasonics=1, num_xrays=1, num_cts=1, total_days=1,
#                            patient_log=patient_log, arrival_rates=arrival_rates, service_times=service_times))
#     env.run()
#     all_run_total_wait_times.append(sum(stats['total_wait_times']))
#
# all_run_total_wait_times_array = np.array(all_run_total_wait_times)
# mean = np.mean(all_run_total_wait_times_array)
# std = np.std(all_run_total_wait_times_array)
# confidence = 0.95
# dof = len(all_run_total_wait_times_array) - 1
# confidence_interval = scipy.stats.t.interval(confidence, dof, loc=mean, scale=std)
# CI_range = (confidence_interval[1] - confidence_interval[0]) / 2
# print(f"Expected waiting time: {mean:.2f} +- {CI_range:.2f}")
# print(f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
#
# # test number of run for stochastics factor
# target_ci_width = 0.5  # stop when CI half-width < 1 hour
# min_runs = 100          # minimum initial runs
# max_runs = 1000        # safety cap to avoid infinite loops
#
# wait_times = []
#
# while True:
#     # Run one more simulation
#     env = simpy.Environment()
#     stats_dict = {}
#     patient_log = []
#     doctor_schedule = chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift)
#     env.process(run_emergency_dept(env, stats=stats_dict, num_triage_nurses=1, doctor_schedule=doctor_schedule,
#                                    num_lab_tests=1, num_ultrasonics=1, num_xrays=1, num_cts=1, total_days=1,
#                                    patient_log=patient_log, arrival_rates=arrival_rates, service_times=service_times))
#     env.run()
#
#     wait_times.append(sum(stats_dict['total_wait_times']))
#     n = len(wait_times)
#
#     if n < min_runs:
#         continue
#
#     # Compute mean and CI half-width
#     mean = np.mean(wait_times)
#     std = np.std(wait_times)
#     std_err = scipy.stats.sem(wait_times)
#     ci = scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=std_err)
#     ci_half_width = (ci[1] - ci[0]) / 2
#
#     print(f"Run {n}: Mean = {mean:.2f}, STD_mean = {std_err: 2f}, sdt = {std}, CI Half-Width = {ci_half_width:.2f}")
#
#     if ci_half_width <= target_ci_width or n >= max_runs:
#         break
#
# print(f"\nFinal Result: {mean:.2f} ± {ci_half_width:.2f} (from {n} replications)")


len_parttime_shift = 3
start_parttime_shift = [8, 9, 10, 14, 15, 19]
alpha = 1.5
beta = 0.8
gamma = 1
chromosome = [3, 4, 2, 0, 1, 0, 0, 0, 0]
penalty_point = fitness_function(chromosome, alpha, beta, gamma, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_triage_nurses=1, num_lab_tests=1, num_ultrasonics=1, num_xrays=1, num_cts=1)
print(penalty_point)