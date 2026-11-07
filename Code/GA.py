import numpy as np
import random

def check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift):  # done
    num_doctors = np.array(chromosome)
    result = True

    # define number of doctors in each hour
    time_slot = np.zeros(24)
    for i, j in enumerate(start_parttime_shift):
        num_parttime_doctors = chromosome[3 + i]  # Get number of doctors for this shift
        for k in range(len_parttime_shift):
            time_slot[j + k] += num_parttime_doctors

    for i in range(24):
        if i in range(6, 14):
            time_slot[i] += chromosome[0]
        elif i in range(14, 22):
            time_slot[i] += chromosome[1]
        else:
            time_slot[i] += chromosome[2]

    # Constraint #2: at least 1 full-time staff in operation
    for i in num_doctors[:3]:
        if i == 0:
            result = False
            break

    # Constraint #3: number of doctor assigned not exceed the number of consultant room
    if result:
        for i in time_slot:
            if i > num_consultant_rooms:
                result = False
                break
    return result

def generate_chromosome(num_fulltime_shifts, num_parttime_shifts,
                        num_consultant_rooms):  # done
    fulltime_schedule = [random.randint(1, num_consultant_rooms) for _ in range(num_fulltime_shifts)]
    parttime_schedule = [random.randint(0, 2) for _ in range(num_parttime_shifts)]  # report that 3 can use 1 fulltime staff instead 3 parttime staff
    chromosome = np.concatenate([fulltime_schedule, parttime_schedule])
    return chromosome

def tournament_selection(population, fitness_scores, tournament_size):  # done # parent selection
    selected = []
    paired_population = list(zip(population, fitness_scores))
    for _ in range(2):
        participants = np.random.choice(len(paired_population), tournament_size, replace=False)
        best_index = min(participants, key=lambda i: paired_population[i][1])
        best_participant = paired_population.pop(best_index)[0]
        selected.append(best_participant)
    return selected[0], selected[1]

def elitism_selection(population, fitness_scores, elitism_size):  # done
    indices = np.argsort(fitness_scores)[:elitism_size]
    selected = [population[i] for i in indices]
    return selected

def one_point_crossover(first_parent, second_parent):  # done # check the crossover point
    # cut point at 3 where separate fulltime and parttime shift
    crossover_point = 3  # np.random.randint(1, len(first_parent)) # check the crossover point

    # swap 2 parts of the cut point
    first_child = np.concatenate((first_parent[:crossover_point], second_parent[crossover_point:]))
    second_child = np.concatenate((second_parent[:crossover_point], first_parent[crossover_point:]))
    return first_child, second_child

def two_point_crossover(first_parent, second_parent):  # done # not check valid and not repair
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

def swap_mutation(chromosome):  # done # not check valid and not repair
    fulltime_or_parttime = random.choice([0, 1])
    if fulltime_or_parttime == 0:
        slot1, slot2 = np.random.choice(range(3), 2, replace=False)
        dummy = chromosome[slot1]
        chromosome[slot1] = chromosome[slot2]
        chromosome[slot2] = dummy
    else:
        slot1, slot2 = np.random.choice(range(3, len(chromosome)), 2, replace=False)
        dummy = chromosome[slot1]
        chromosome[slot1] = chromosome[slot2]
        chromosome[slot2] = dummy
    return chromosome

def repair_chromosome(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift):  # done
    is_valid, invalid_constraints = check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
    while not is_valid:
        # check constraint #1
        if invalid_constraints[0] == 1:
            for i in range(3):
                if chromosome[i] == 0:
                    chromosome[i] += 1

        # check constraint #2:
        if invalid_constraints[1] == 1:
            time_slot = np.zeros(24)
            parttime_contribution = np.zeros((len(start_parttime_shift), 24))

            for i, j in enumerate(start_parttime_shift):
                num_parttime_doctors = chromosome[3 + i]  # Get number of doctors for this shift
                for k in range(len_parttime_shift):
                    t = j + k
                    time_slot[t] += num_parttime_doctors
                    parttime_contribution[i][t] = 1

            for _ in range(24):
                if _ < 8:
                    time_slot[_] += chromosome[0]
                elif _ < 16:
                    time_slot[_] += chromosome[1]
                else:
                    time_slot[_] += chromosome[2]

            for t in range(24):
                while time_slot[t] > num_consultant_rooms:
                    reduced = False
                    for i in range(len(start_parttime_shift)):
                        if parttime_contribution[i][t] == 1 and chromosome[3 + i] > 0:
                            chromosome[3 + i] -= 1
                            for k in range(len_parttime_shift):
                                affected_slot = start_parttime_shift[i] + k
                                time_slot[affected_slot] -= 1
                            reduced = True
                            break
                    if not reduced:
                        break

        # recheck
        is_valid, invalid_constraints = check_validation(chromosome, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
    return chromosome

def replacement(population, penalty_points, first_child, second_child, first_penalty_point, second_penalty_point):  # done
    combined = list(zip(population, penalty_points))
    combined.append((first_child, first_penalty_point))
    combined.append((second_child, second_penalty_point))
    combined.sort(key=lambda x: x[1])
    combined = combined[:len(population)]
    new_population = [ind for ind, _ in combined]
    new_penalty_points = [pen for _, pen in combined]
    return new_population, new_penalty_points


"""def genetic_algorithm_run(num_generation, population_size, tournament_size, crossover_probability, mutation_probability,
                          num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms, len_parttime_shift,
                          start_parttime_shift, alpha, beta, gamma, run_id):  # done
    population = []
    fitness_scores = []
    generation_data = []

    # generate initial population and get fitness scores
    for i in range(population_size):
        while True:
            chromosome = generate_chromosome(num_fulltime_shifts, num_parttime_shifts, num_consultant_rooms)
            result, invalid_constraint = check_validation(chromosome, num_consultant_rooms, len_parttime_shift,
                                                          start_parttime_shift)
            if result == True:
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
        best_score.append(fitness_scores[0])

        print("Gen: ", i, " best chromosome: ", population[0], " score: ", fitness_scores[0])

        # write value
        generation_data.append({
            "Generation": i,
            "Best Score": fitness_scores[0],
            "Best Chromosome": population[0]
        })

    generation_df = pd.DataFrame(generation_data)
    generation_df.to_excel("Phase1_GA_Result.xlsx", sheet_name=f"Best Genes (Run {run_id})", index=False)

    final_best_chromosome = best_chromosome[-1]
    final_best_score = best_score[-1]

    return final_best_chromosome, final_best_score, best_chromosome, best_score, population, fitness_scores"""