import numpy as np
import random

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
    # Constraint #3: physicians on duty <= consulting rooms
    # (fixed 2026-11-07: the original rejected hours with exactly R physicians)
    if result:
        for i in time_slot_doctors:
            if i > num_consultant_rooms:
                result = False
                break
    # Constraint #4: full-time nurses >= full-time physicians in every shift
    # (fixed 2026-11-07: the original required strictly more nurses)
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
    # Cut between the full-time and part-time blocks.
    # (fixed 2026-11-07: the original returned two identical children)
    crossover_point = 3
    first_child = np.hstack([first_parent[:, :crossover_point], second_parent[:, crossover_point:]])
    second_child = np.hstack([second_parent[:, :crossover_point], first_parent[:, crossover_point:]])
    return first_child, second_child


def two_point_crossover(first_parent, second_parent):  # done
    # random 2 cut points
    cutpoint1, cutpoint2 = np.random.choice(range(1, len(first_parent)-1), 2, replace=False)
    if cutpoint1 > cutpoint2:
        cutpoint1, cutpoint2 = cutpoint2, cutpoint1

    # swap the middle part between 2 cut points
    first_child = np.concatenate(
        (first_parent[:cutpoint1], second_parent[cutpoint1:cutpoint2], first_parent[cutpoint2:]))
    second_child = np.concatenate(
        (second_parent[:cutpoint1], first_parent[cutpoint1:cutpoint2], second_parent[cutpoint2:]))
    return first_child, second_child

def swap_mutation(chromosome):
    # 0: swap in the nurse row, 1: swap in the physician row, 2: swap the same pair in both rows.
    # (fixed 2026-11-07: the original used if/if/else, so case 0 also applied case 2)
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

"""def genetic_algorithm_run(num_generation, population_size, tournament_size, crossover_probability, mutation_probability,
                          num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms, len_parttime_shift,
                          start_parttime_shift, alpha, beta, gamma):  # done
    population = []
    fitness_scores = []
    generation_data = []

    # generate initial population and get fitness scores
    for i in range(population_size):
        while True:
            chromosome = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)
            result = check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if result:
                break
        population.append(chromosome)
        score = fitness_function(chromosome, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)
        fitness_scores.append(score)
        print("Pop: ", i, " chromosome ", chromosome, " point: ", score)

    fitness_scores = np.array(fitness_scores)
    population = np.array(population)

    population = population[fitness_scores.argsort()]
    fitness_scores = fitness_scores[fitness_scores.argsort()]
    best_chromosome = []
    best_score = []

    # reproduce new chromosomes in new generation
    for i in range(num_generation):
        # selection
        parent1, parent2 = tournament_selection(population, fitness_scores, tournament_size)
        print("parent1: ", parent1, " parent2: ", parent2)

        # crossover
        if random.random() <= crossover_probability:
            child1, child2 = one_point_crossover(parent1, parent2)
        else:
            child1 = parent1
            child2 = parent2

        # mutation
        if random.random() <= mutation_probability:
            child1 = swap_mutation(child1)
        if random.random() <= mutation_probability:
            child2 = swap_mutation(child2)

        # check valid:
        while True:
            result_child1 = check_validation(child1, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if result_child1:
                break
            child1 = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        while True:
            result_child2 = check_validation(child2, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
            if result_child2:
                break
            child2 = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)

        # penalty evaluation
        score1 = fitness_function(child1, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)
        score2 = fitness_function(child2, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)
        print("child1: ", child1, "_", score1, " child2: ", child2, "_", score2)

        # replacement
        population, fitness_scores = replacement(population, fitness_scores, child1, child2, score1, score2)

        # store value
        best_chromosome.append(population[0])
        best_score.append(fitness_scores[0])

        print("Gen: ", i, " best chromosome: ", population[0], " score: ", fitness_scores[0])

        # write value
        generation_data.append({
            "Generation": i,
            "Best Score": fitness_scores[0],
            "Best Chromosome": population[0]
        })

    generation_df = pd.DataFrame(generation_data)
    generation_df.to_excel("Phase1_GA_Result.xlsx", sheet_name=f"Best Genes (Run 0)", index=False)

    final_best_chromosome = best_chromosome[-1]
    final_best_score = best_score[-1]

    return final_best_chromosome, final_best_score, best_chromosome, best_score, population, fitness_scores"""
