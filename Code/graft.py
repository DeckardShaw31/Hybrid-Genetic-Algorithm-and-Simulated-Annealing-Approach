"""Phase-1 HGASA (nurses and physicians), monolithic version.

Patched 2026-11-07 with the same fixes as GA_phyNnurse.py / SA_phyNnurse.py:
- room constraint (<= R) and full-time nurse constraint (>=);
- crossover second child; swap-mutation if/elif;
- attempt caps in neighbourhood structures 3 and 4.
The simulation is the corrected shared-pool DES in DES_phyNnurse.py.
For the re-runs, use Phase1_rerun/run_phase1.py instead.
"""
import numpy as np
import pandas as pd
import random
import simpy
import time
import ast
from DES_phyNnurse import run_emergency_dept

def chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift):
    num_nurses = chromosome[0].copy()
    num_doctors = chromosome[1].copy()
    time_slot_nurses = np.zeros(24)
    time_slot_doctors = np.zeros(24)
    schedule_nurses = {}
    schedule_doctors = {}

    # Nurse schedule
    for i, j in enumerate(start_parttime_shift):
        num_parttime_nurses = num_nurses[3 + i]
        for k in range(len_parttime_shift):
            time_slot_nurses[j + k] += num_parttime_nurses
    for i in range(24):
        if i in range(6, 14):
            time_slot_nurses[i] += num_nurses[0]
        elif i in range(14, 22):
            time_slot_nurses[i] += num_nurses[1]
        else:
            time_slot_nurses[i] += num_nurses[2]
    for i in range(24):
        schedule_nurses.update({i: time_slot_nurses[i]})

    # Doctor schedule
    for i, j in enumerate(start_parttime_shift):
        num_parttime_doctors = num_doctors[3 + i]
        for k in range(len_parttime_shift):
            time_slot_doctors[j + k] += num_parttime_doctors
    for i in range(24):
        if i in range(6, 14):
            time_slot_doctors[i] += num_doctors[0]
        elif i in range(14, 22):
            time_slot_doctors[i] += num_doctors[1]
        else:
            time_slot_doctors[i] += num_doctors[2]
    for i in range(24):
        schedule_doctors.update({i: time_slot_doctors[i]})

    return schedule_nurses, schedule_doctors

def fitness_function(chromosome, alpha, beta, gamma, delta, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_lab_tests, num_ultrasonics, num_xrays, num_cts):
    num_nurses = chromosome[0].copy()
    num_doctors = chromosome[1].copy()
    fulltime_nurse_working_hour = 0
    parttime_nurse_working_hour = 0
    fulltime_doctor_working_hour = 0
    parttime_doctor_working_hour = 0

    # calculate first term
    for i in num_nurses[:3]:
        fulltime_nurse_working_hour += i * 8
    # calculate second term
    for i in num_nurses[3:]:
        parttime_nurse_working_hour += i * len_parttime_shift
    # calculate third term
    for i in num_doctors[:3]:
        fulltime_doctor_working_hour += i * 8
    # calculate fourth term
    for i in num_doctors[3:]:
        parttime_doctor_working_hour += i * len_parttime_shift

    # run simulation for stochastics input
    all_run_total_wait_times = []
    all_run_total_delay_times = []
    for i in range(500):
        env = simpy.Environment()
        stats = {}
        patient_log = []
        nurse_schedule, doctor_schedule = chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift)
        env.process(run_emergency_dept(env, stats=stats, nurse_schedule=nurse_schedule, doctor_schedule=doctor_schedule, num_lab_tests=num_lab_tests, num_ultrasonics=num_ultrasonics, num_xrays=num_xrays, num_cts=num_cts, total_days=1, patient_log=patient_log, arrival_rates=arrival_rates, service_times=service_times))
        env.run()
        all_run_total_wait_times.append(sum(stats['total_wait_times']))
        total_delaying_times_with_examination = [i - 2 for i in stats['total_wait_times_with_examination'] if i > 2]
        total_delaying_times_without_examination = [i - 0.5 for i in stats['total_wait_times_without_examination'] if i > 0.5]
        all_run_total_delay_times.append(sum(total_delaying_times_with_examination + total_delaying_times_without_examination))

    # calculate fifth term
    expected_waiting_times = sum(all_run_total_wait_times)/len(all_run_total_wait_times)
    # calculate sixth term
    expected_delaying_times = sum(all_run_total_delay_times) / len(all_run_total_delay_times)

    # calculate penalty
    penalty_point = fulltime_doctor_working_hour + alpha*parttime_doctor_working_hour + beta*(fulltime_nurse_working_hour + alpha*parttime_nurse_working_hour) + gamma*expected_waiting_times + delta*expected_delaying_times

    element = [fulltime_doctor_working_hour, parttime_doctor_working_hour, fulltime_nurse_working_hour, parttime_nurse_working_hour, expected_waiting_times, expected_delaying_times]
    return penalty_point, element

def check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift):
    num_nurses = chromosome[0].copy()
    num_doctors = chromosome[1].copy()
    result = True

    # define number of nurses in each hour
    time_slot_nurses = np.zeros(24)
    for i, j in enumerate(start_parttime_shift):
        num_parttime_nurses = num_nurses[3 + i]
        for k in range(len_parttime_shift):
            time_slot_nurses[j + k] += num_parttime_nurses
    for i in range(24):
        if i in range(6, 14):
            time_slot_nurses[i] += num_nurses[0]
        elif i in range(14, 22):
            time_slot_nurses[i] += num_nurses[1]
        else:
            time_slot_nurses[i] += num_nurses[2]
    # define number of doctors in each hour
    time_slot_doctors = np.zeros(24)
    for i, j in enumerate(start_parttime_shift):
        num_parttime_doctors = num_doctors[3 + i]
        for k in range(len_parttime_shift):
            time_slot_doctors[j + k] += num_parttime_doctors
    for i in range(24):
        if i in range(6, 14):
            time_slot_doctors[i] += num_doctors[0]
        elif i in range(14, 22):
            time_slot_doctors[i] += num_doctors[1]
        else:
            time_slot_doctors[i] += num_doctors[2]

    # Constraint #2:
    for i in num_doctors[:3]:
        if i == 0:
            result = False
            break
    for i in num_nurses[:3]:
        if i == 0:
            result = False
            break
    # Constraint #3:
    if result:
        for i in time_slot_doctors:
            if i > num_consultant_rooms:
                result = False
                break
    # Constraint #4:
    if result:
        for i in range(3):
            if num_nurses[i] < num_doctors[i]:
                result = False
                break
    # Constraint #5:
    if result:
        for i in range(24):
            if time_slot_nurses[i] < time_slot_doctors[i]:
                result = False
                break
    return result

# GENETIC ALGORITHM GA
def generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms):
    fulltime_schedule_doctors = [random.randint(1, num_consultant_rooms) for _ in range(num_fulltime_shifts)]
    parttime_schedule_doctors = [random.choices([0, 1], weights=[0.6, 0.4])[0] for _ in range(num_parttime_shifts)]
    fulltime_schedule_nurses = [random.randint(fulltime_schedule_doctors[_], fulltime_schedule_doctors[_]+2) for _ in range(num_fulltime_shifts)]
    parttime_schedule_nurses = [random.choices([0, 1], weights=[0.6, 0.4])[0] for _ in range(num_parttime_shifts)]
    chromosome = np.array([fulltime_schedule_nurses + parttime_schedule_nurses, fulltime_schedule_doctors + parttime_schedule_doctors])
    return chromosome

def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    paired_population = list(zip(population, fitness_scores))
    for _ in range(2):
        participants = np.random.choice(len(paired_population), tournament_size, replace=False)
        best_index = min(participants, key=lambda i: paired_population[i][1])
        best_participant = paired_population.pop(best_index)[0]
        selected.append(best_participant)
    return selected[0], selected[1]

def one_point_crossover(first_parent, second_parent):
    crossover_point = 3
    first_child = np.hstack([first_parent[:, :crossover_point], second_parent[:, crossover_point:]])
    second_child = np.hstack([second_parent[:, :crossover_point], first_parent[:, crossover_point:]])
    return first_child, second_child

def swap_mutation(chromosome):
    chromosome = chromosome.copy()
    nurse_or_physician = random.choice(range(3))
    fulltime_or_parttime = random.choice([0, 1])
    if nurse_or_physician == 0:
        if fulltime_or_parttime == 0:
            slot1, slot2 = np.random.choice(range(3), 2, replace=False)
            dummy = chromosome[0, slot1]
            chromosome[0, slot1] = chromosome[0, slot2]
            chromosome[0, slot2] = dummy
        else:
            slot1, slot2 = np.random.choice(range(3, np.size(chromosome, 1)), 2, replace=False)
            dummy = chromosome[0, slot1]
            chromosome[0, slot1] = chromosome[0, slot2]
            chromosome[0, slot2] = dummy
    elif nurse_or_physician == 1:
        if fulltime_or_parttime == 0:
            slot1, slot2 = np.random.choice(range(3), 2, replace=False)
            dummy = chromosome[1, slot1]
            chromosome[1, slot1] = chromosome[1, slot2]
            chromosome[1, slot2] = dummy
        else:
            slot1, slot2 = np.random.choice(range(3, np.size(chromosome, 1)), 2, replace=False)
            dummy = chromosome[1, slot1]
            chromosome[1, slot1] = chromosome[1, slot2]
            chromosome[1, slot2] = dummy
    else:
        if fulltime_or_parttime == 0:
            slot1, slot2 = np.random.choice(range(3), 2, replace=False)
            dummy = chromosome[0, slot1]
            chromosome[0, slot1] = chromosome[0, slot2]
            chromosome[0, slot2] = dummy
            dummy1 = chromosome[1, slot1]
            chromosome[1, slot1] = chromosome[1, slot2]
            chromosome[1, slot2] = dummy1
        else:
            slot1, slot2 = np.random.choice(range(3, np.size(chromosome, 1)), 2, replace=False)
            dummy = chromosome[0, slot1]
            chromosome[0, slot1] = chromosome[0, slot2]
            chromosome[0, slot2] = dummy
            dummy1 = chromosome[1, slot1]
            chromosome[1, slot1] = chromosome[1, slot2]
            chromosome[1, slot2] = dummy1
    return chromosome

def replacement(population, penalty_points, first_child, second_child, first_penalty_point, second_penalty_point, elements, element1, element2):  # done
    combined = list(zip(population, penalty_points, elements))
    combined.append((first_child, first_penalty_point, element1))
    combined.append((second_child, second_penalty_point, element2))
    combined.sort(key=lambda x: x[1])
    combined = combined[:len(population)]
    new_population = [ind for ind, _, _ in combined]
    new_penalty_points = [pen for _, pen, _ in combined]
    new_elements = [ele for _, _, ele in combined]
    return new_population, new_penalty_points, new_elements

# SIMULATED ANNEALING SA
# random decrease 1 unit
def neighbourhood_structure1(candidate):
    for _ in range(20):
        a = np.random.choice(range(np.size(candidate, 0)))
        b = np.random.choice(range(np.size(candidate, 1)))
        if b in range(3):
            if candidate[a, b] > 1:
                candidate[a, b] -= 1
                break
        else:
            if candidate[a, b] > 0:
                candidate[a, b] -= 1
                break
    return candidate

# random increase 1 unit
def neighbourhood_structure2(candidate):
    a = np.random.choice(range(np.size(candidate, 0)))
    b = np.random.choice(range(np.size(candidate, 1)))
    candidate[a, b] += 1
    return candidate

# change 1 full-time to corresponding part-time shift
def neighbourhood_structure3(candidate, start_parttime_shift, len_parttime_shift):
    check = False
    attempts = 0
    while not check:
        attempts += 1
        if attempts > 1000:
            return candidate
        nurse_or_physician = random.choice([0, 1])
        num_staff = candidate[nurse_or_physician].copy()
        full_gene = np.random.choice(range(3))
        part_gene = np.random.choice(range(3, len(num_staff)))
        time_slot = np.zeros(24)
        if full_gene == 0:
            for i in range(6, 14):
                time_slot[i] += 1
        elif full_gene == 1:
            for i in range(14, 22):
                time_slot[i] += 1
        else:
            for i in range(0, 6):
                time_slot[i] += 1
            for i in range(22, 24):
                time_slot[i] += 1

        for i, j in enumerate(start_parttime_shift):
            if i == part_gene - 3:
                for k in range(len_parttime_shift):
                    time_slot[j + k] += 1
        check = np.any(time_slot == 2) and num_staff[full_gene] > 1
    num_staff[full_gene] -= 1
    num_staff[part_gene] += 1
    candidate[nurse_or_physician] = num_staff.copy()
    return candidate

# change 1 part-time to corresponding full-time shift
def neighbourhood_structure4(candidate, start_parttime_shift, len_parttime_shift):
    check = False
    attempts = 0
    while not check:
        attempts += 1
        if attempts > 1000:
            return candidate
        nurse_or_physician = random.choice([0, 1])
        num_staff = candidate[nurse_or_physician].copy()
        full_gene = np.random.choice(range(3))
        part_gene = np.random.choice(range(3, len(num_staff)))
        time_slot = np.zeros(24)
        if full_gene == 0:
            for i in range(6, 14):
                time_slot[i] += 1
        elif full_gene == 1:
            for i in range(14, 22):
                time_slot[i] += 1
        else:
            for i in range(0, 6):
                time_slot[i] += 1
            for i in range(22, 24):
                time_slot[i] += 1
        for i, j in enumerate(start_parttime_shift):
            if i == part_gene - 3:
                for k in range(len_parttime_shift):
                    time_slot[j + k] += 1
        check = np.any(time_slot == 2) and num_staff[part_gene] > 0
    num_staff[full_gene] += 1
    num_staff[part_gene] -= 1
    candidate[nurse_or_physician] = num_staff.copy()
    return candidate

def hybrid_genetic_algorithm_simulated_annealing_run(num_generation, population_size, tournament_size, crossover_probability, mutation_probability, initial_temperature, final_temperature, cooling_rate, num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms, len_parttime_shift, start_parttime_shift, alpha, beta, gamma, delta, num_lab_tests, num_ultrasonics, num_xrays, num_cts, run_id, arrival_rates, service_times):
    # GENETIC ALGORITHM GA
    population = []
    fitness_scores = []
    elements = []
    generation_data = []
    # generate initial population and get fitness scores
    for i in range(population_size):
        while True:
            chromosome = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)
            is_valid_chromosome = check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_chromosome:
                break
        population.append(chromosome)
        score, element = fitness_function(chromosome, alpha, beta, gamma, delta, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_lab_tests, num_ultrasonics, num_xrays, num_cts)
        fitness_scores.append(score)
        elements.append(element)

    fitness_scores = np.array(fitness_scores)
    population = np.array(population)
    elements = np.array(elements)

    population = population[fitness_scores.argsort()]
    fitness_scores = fitness_scores[fitness_scores.argsort()]
    elements = elements[fitness_scores.argsort()]
    best_chromosome = []
    best_score = []
    best_element = []

    # reproduce new chromosomes in new generation
    for i in range(num_generation):
        # selection
        parent1, parent2 = tournament_selection(population, fitness_scores, tournament_size)
        # crossover
        if random.random() <= crossover_probability:
            child1, child2 = one_point_crossover(parent1, parent2)
        else:
            child1 = parent1
            child2 = parent2
        # check valid:
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
        # check valid:
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
        score1, element1 = fitness_function(child1, alpha, beta, gamma, delta, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_lab_tests, num_ultrasonics, num_xrays, num_cts)
        score2, element2 = fitness_function(child2, alpha, beta, gamma, delta, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_lab_tests, num_ultrasonics, num_xrays, num_cts)
        # replacement
        population, fitness_scores, elements = replacement(population, fitness_scores, child1, child2, score1, score2, elements, element1, element2)

        # store value
        best_chromosome.append(population[0])
        best_score.append(fitness_scores[0])
        best_element.append(elements[0])
        # write value
        generation_data.append({
            "Generation": i,
            "Best Score": fitness_scores[0],
            "Best Chromosome": population[0],
            "Physician Fulltime": elements[0][0],
            "Physician Parttime": elements[0][1],
            "Nurse Fulltime": elements[0][2],
            "Nurse Parttime": elements[0][3],
            "Waiting Time": elements[0][4],
            "Delaying Time": elements[0][5],
            "Temperature": None
        })

    ga_final_best_chromosome = best_chromosome[-1]
    ga_final_best_score = best_score[-1]
    ga_final_best_element = best_element[-1]

    # SIMULATED ANNEALING SA
    temperature = initial_temperature
    current_candidate = ga_final_best_chromosome.copy()
    best_candidate = ga_final_best_chromosome.copy()
    current_penalty_point = ga_final_best_score
    best_penalty_point = ga_final_best_score
    current_element = ga_final_best_element.copy()
    best_sa_element = ga_final_best_element.copy()

    iteration = 0
    neighbourhood_structure_no = 4
    while temperature >= final_temperature:
        neighbourhood_structure = np.random.randint(neighbourhood_structure_no)
        new_candidate = current_candidate.copy()
        if neighbourhood_structure == 0:
            new_candidate = neighbourhood_structure1(new_candidate)
        elif neighbourhood_structure == 1:
            new_candidate = neighbourhood_structure2(new_candidate)
        elif neighbourhood_structure == 2:
            new_candidate = neighbourhood_structure3(new_candidate, start_parttime_shift, len_parttime_shift)
        else:
            new_candidate = neighbourhood_structure4(new_candidate, start_parttime_shift, len_parttime_shift)
        while True:
            is_valid_candidate = check_validation(new_candidate, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if is_valid_candidate:
                break
            new_candidate = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        new_penalty_point, new_element = fitness_function(new_candidate, alpha, beta, gamma, delta, len_parttime_shift, start_parttime_shift, arrival_rates, service_times, num_lab_tests, num_ultrasonics, num_xrays, num_cts)
        difference = new_penalty_point - current_penalty_point

        if difference <= 0 or np.random.random() < np.exp((-1 * difference) / temperature):
            current_candidate = new_candidate.copy()
            current_penalty_point = new_penalty_point.copy()
            current_element = new_element.copy()
        if current_penalty_point < best_penalty_point:
            best_candidate = current_candidate.copy()
            best_penalty_point = current_penalty_point.copy()
            best_sa_element = current_element.copy()

        generation_data.append({
            "Generation": iteration,
            "Best Score": best_penalty_point,
            "Best Chromosome": best_candidate,
            "Physician Fulltime": best_sa_element[0],
            "Physician Parttime": best_sa_element[1],
            "Nurse Fulltime": best_sa_element[2],
            "Nurse Parttime": best_sa_element[3],
            "Waiting Time": best_sa_element[4],
            "Delaying Time": best_sa_element[5],
            "Temperature": temperature,
        })

        temperature *= cooling_rate
        iteration += 1
    generation_df = pd.DataFrame(generation_data)
    generation_df.to_excel(writer, sheet_name=f"Best Genes (Run {run_id})", index=False)
    return generation_df, best_candidate, best_penalty_point


# RUN MODEL
for no_data in range(1, 3):
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
        df_arrival_rates = pd.read_excel(file_path, sheet_name=f"Arrival rates Day {day}")
        arrival_rates = df_arrival_rates.iloc[:, 0].tolist()
        output_file = f"Phase1_HGASA_Result Day {day} Data {no_data} (nurse and physician).xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            run_time = []
            for run_id in range(1, 11):
                print(f"Starting Data {no_data} Day {day} Run {run_id}")
                start_run_time = time.perf_counter()
                hybrid_genetic_algorithm_simulated_annealing_run(
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
                    delta=delta,
                    num_lab_tests=num_lab_tests,
                    num_ultrasonics=num_ultrasonics,
                    num_xrays=num_xrays,
                    num_cts=num_cts,
                    run_id=run_id,
                    arrival_rates=arrival_rates,
                    service_times=service_times)
                end_run_time = time.perf_counter()
                run_time.append({
                    "Run": run_id,
                    "Run time": end_run_time - start_run_time
                })
                runtime_df = pd.DataFrame(run_time)
                runtime_df.to_excel(writer, sheet_name="Runtime", index=False)
