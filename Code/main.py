import random
import numpy as np
import pandas as pd
import ast
import time
from GA import check_validation, generate_chromosome, tournament_selection, one_point_crossover, swap_mutation, replacement
from SA import neighbourhood_structure1, neighbourhood_structure2, neighbourhood_structure3, neighbourhood_structure4
from penalty import fitness_function

def hybrid_genetic_algorithm_simulated_annealing_run(num_generation, population_size, tournament_size, crossover_probability,
                                                     mutation_probability, initial_temperature, final_temperature, cooling_rate,
                                                     num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms,
                                                     len_parttime_shift, start_parttime_shift, alpha, beta, gamma,
                                                     num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays,
                                                     num_cts, run_id, arrival_rates, service_times):
    # GENETIC ALGORITHM GA
    population = []
    fitness_scores = []
    generation_data = []

    # generate initial population and get fitness scores
    for i in range(population_size):
        print("Pop: ", i)
        while True:
            chromosome = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)
            is_valid_chromosome = check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_chromosome:
                break
        population.append(chromosome)
        score = fitness_function(chromosome, alpha, beta, gamma, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays, num_cts)
        fitness_scores.append(score)

    fitness_scores = np.array(fitness_scores)
    population = np.array(population)

    population = population[fitness_scores.argsort()]
    fitness_scores = fitness_scores[fitness_scores.argsort()]
    best_chromosome = []
    best_score = []

    # reproduce new chromosomes in new generation
    for i in range(num_generation):
        print("Gen: ", i)
        # selection
        parent1, parent2 = tournament_selection(population, fitness_scores, tournament_size)

        # crossover
        if random.random() <= crossover_probability:
            child1, child2 = one_point_crossover(parent1, parent2)
        else:
            child1 = parent1
            child2 = parent2

        # check valid
        while True:
            is_valid_child1 = check_validation(child1, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_child1:
                break
            child1 = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        while True:
            is_valid_child2 = check_validation(child2, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_child2:
                break
            child2 = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        # mutation
        if random.random() <= mutation_probability:
            child1 = swap_mutation(child1)
        if random.random() <= mutation_probability:
            child2 = swap_mutation(child2)

        # check valid
        while True:
            is_valid_child1 = check_validation(child1, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_child1:
                break
            child1 = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        while True:
            is_valid_child2 = check_validation(child2, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_child2:
                break
            child2 = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        # penalty evaluation
        score1 = fitness_function(child1, alpha, beta, gamma, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays, num_cts)
        score2 = fitness_function(child2, alpha, beta, gamma, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays, num_cts)

        # replacement
        population, fitness_scores = replacement(population, fitness_scores, child1, child2, score1, score2)

        # store value
        best_chromosome.append(population[0])
        best_score.append(fitness_scores[0])

        # write value
        generation_data.append({
            "Generation": i,
            "Best Score": fitness_scores[0],
            "Best Chromosome": population[0],
            "Temperature": None
        })

    ga_final_best_chromosome = best_chromosome[-1]
    ga_final_best_score = best_score[-1]

    # SIMULATED ANNEALING SA
    temperature = initial_temperature

    current_candidate = ga_final_best_chromosome.copy()
    best_candidate = ga_final_best_chromosome.copy()
    current_penalty_point = ga_final_best_score
    best_penalty_point = ga_final_best_score

    iteration = 0
    neighbourhood_structure_no = 4

    while temperature >= final_temperature:
        print("Iteration: ", iteration)
        neighbourhood_structure = np.random.randint(neighbourhood_structure_no)
        new_candidate = current_candidate.copy()

        if neighbourhood_structure == 0:
            new_candidate = neighbourhood_structure1(new_candidate)
        elif neighbourhood_structure == 1:
            new_candidate = neighbourhood_structure2(new_candidate)
        elif neighbourhood_structure == 2:
            new_candidate = neighbourhood_structure3(new_candidate)
        else:
            new_candidate = neighbourhood_structure4(new_candidate)

        while True:
            is_valid_candidate = check_validation(new_candidate, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_candidate:
                break
            new_candidate = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        new_penalty_point = fitness_function(new_candidate, alpha, beta, gamma, len_parttime_shift, start_parttime_shift,
                                             arrival_rates, service_times, num_triage_nurses, num_lab_tests, num_ultrasonics,
                                             num_xrays, num_cts)
        difference = new_penalty_point - current_penalty_point

        if difference < 0 or np.random.random() < np.exp((-1 * difference) / temperature):
            current_candidate = new_candidate.copy()
            current_penalty_point = new_penalty_point.copy()

        if current_penalty_point < best_penalty_point:
            best_candidate = current_candidate.copy()
            best_penalty_point = current_penalty_point.copy()

        generation_data.append({
            "Generation": iteration,
            "Best Score": best_penalty_point,
            "Best Chromosome": best_candidate,
            "Temperature": temperature
        })

        temperature *= cooling_rate
        iteration += 1

    generation_df = pd.DataFrame(generation_data)
    generation_df.to_excel(writer, sheet_name=f"Best Genes (Run {run_id})", index=False)
    print("best_candidate: ", best_candidate, "best_penalty_point: ", best_penalty_point)
    return generation_data, best_candidate, best_penalty_point

# INPUT OUTPUT
for no_data in range(1, 3):
    if no_data == 2:
        continue

    file_path = f"Data{no_data}.xlsx"

    df_parameters = pd.read_excel(file_path, sheet_name="Parameters")
    for _, row in df_parameters.iterrows():
        key = row['Parameter']
        value = row['Value']
        value = ast.literal_eval(str(value))
        globals()[key] = value

    df_service_times = pd.read_excel(file_path, sheet_name="Service times")
    service_times = df_service_times.iloc[:, 0].tolist()

    for day in range(7):
        if day > 0:
            continue
        df_arrival_rates = pd.read_excel(file_path, sheet_name=f"Arrival rates Day {day}")
        arrival_rates = df_arrival_rates.iloc[:, 0].tolist()
        output_file = f"Phase1_HGASA_Result Day {day} Data {no_data} (physician only).xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            result = []

            for run_id in range(1, 11):
                print(f"Starting Day {day} Run {run_id}")
                start_run_time = time.perf_counter()
                generation_data, best_candidate, best_penalty_point = hybrid_genetic_algorithm_simulated_annealing_run(
                    num_generation=num_generation,
                    population_size=population_size,
                    tournament_size=tournament_size,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    initial_temperature=initial_temperature,
                    final_temperature=final_temperature,
                    cooling_rate=cooling_rate,
                    num_fulltime_shifts=num_fulltime_shifts,
                    num_parttime_shifts=num_parttime_shifts,
                    num_consultant_rooms=num_consultant_rooms,
                    len_parttime_shift=len_parttime_shift,
                    start_parttime_shift=start_parttime_shift,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    num_triage_nurses=num_triage_nurses,
                    num_lab_tests=num_lab_tests,
                    num_ultrasonics=num_ultrasonics,
                    num_xrays=num_xrays,
                    num_cts=num_cts,
                    run_id=run_id,
                    arrival_rates=arrival_rates,
                    service_times=service_times)
                end_run_time = time.perf_counter()
                result.append({
                    "Run": run_id,
                    "Run time": end_run_time - start_run_time,
                    "Objective": best_penalty_point
                })
                result_df = pd.DataFrame(result)
                result_df.to_excel(writer, sheet_name="Runtime", index=False)
                print(f"Runtime: Day {day} Run {run_id}: ", end_run_time - start_run_time)

# # OUTPUT
# file_path = f"Data1.xlsx"
#
# df_service_times = pd.read_excel(file_path, sheet_name="Service times")
# service_times = df_service_times.iloc[:, 0].tolist()
#
# df_arrival_rates = pd.read_excel(file_path, sheet_name=f"Arrival rates Day 0")
# arrival_rates = df_arrival_rates.iloc[:, 0].tolist()
#
# for para in range(5, 7):
#     output_file = f"Sensitivity Analysis Parameter {para}.xlsx"
#     with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
#         run_time = []
#         for change in range(6):
#             df_parameters = pd.read_excel(file_path, sheet_name=f"Parameters {para}-{change}")
#             for _, row in df_parameters.iterrows():
#                 key = row['Parameter']
#                 value = row['Value']
#                 value = ast.literal_eval(str(value))
#                 globals()[key] = value
#
#             for run_id in range(10):
#                 print(f"Parameter {para}-{change}-{run_id}")
#                 start_run_time = time.perf_counter()
#
#                 hybrid_genetic_algorithm_simulated_annealing_run(
#                     num_generation=num_generation,
#                     population_size=population_size,
#                     tournament_size=tournament_size,
#                     crossover_probability=crossover_probability,
#                     mutation_probability=mutation_probability,
#                     initial_temperature=initial_temperature,
#                     final_temperature=final_temperature,
#                     cooling_rate=cooling_rate,
#                     num_fulltime_shifts=num_fulltime_shifts,
#                     num_parttime_shifts=num_parttime_shifts,
#                     num_consultant_rooms=num_consultant_rooms,
#                     len_parttime_shift=len_parttime_shift,
#                     start_parttime_shift=start_parttime_shift,
#                     alpha=alpha,
#                     beta=beta,
#                     gamma=gamma,
#                     num_triage_nurses=num_triage_nurses,
#                     num_lab_tests=num_lab_tests,
#                     num_ultrasonics=num_ultrasonics,
#                     num_xrays=num_xrays,
#                     num_cts=num_cts,
#                     run_id=run_id,
#                     arrival_rates=arrival_rates,
#                     service_times=service_times,
#                     change=change)
#                 end_run_time = time.perf_counter()
#                 run_time.append({
#                     "Change": change,
#                     "Run": run_id,
#                     "Run time": end_run_time - start_run_time
#                 })
#                 runtime_df = pd.DataFrame(run_time)
#                 runtime_df.to_excel(writer, sheet_name="Runtime", index=False)
#                 print(f"Runtime Parameter {para}-{run_id} ", end_run_time - start_run_time)

"""best_scores = [entry["Best Score"] for entry in generation_data]
plt.figure(figsize=(10, 6))
plt.plot(best_scores)
plt.grid(True)
plt.tight_layout()
plt.show()"""
