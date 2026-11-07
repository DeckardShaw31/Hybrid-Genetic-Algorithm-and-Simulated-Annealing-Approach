import numpy as np
import random

# random decrease 1 unit
def neighbourhood_structure1(candidate):
    # Fixed 2026-11-07. The original treated only genes 0-1 as full-time (the night shift, gene 2,
    # could drop to 0), did not stop after decreasing a part-time gene (so it kept decreasing),
    # and could loop forever.
    for _ in range(1000):
        a = random.randint(0, len(candidate) - 1)
        if a in range(0, 3):
            if candidate[a] > 1:
                candidate[a] -= 1
                break
        else:
            if candidate[a] > 0:
                candidate[a] -= 1
                break
    return candidate

# random increase 1 unit
def neighbourhood_structure2(candidate):
    a = random.randint(0, len(candidate)-1)
    candidate[a] += 1
    return candidate

# swap 2 random slot of fulltime shift
def neighbourhood_structure3(candidate):
    slot1, slot2 = np.random.choice(range(3), 2, replace=False)
    if candidate[slot1] != candidate[slot2]:
        dummy = candidate[slot1]
        candidate[slot1] = candidate[slot2]
        candidate[slot2] = dummy
    return candidate

# swap 2 random slot of parttime shift
def neighbourhood_structure4(candidate):
    slot1, slot2 = np.random.choice(range(3, len(candidate)), 2, replace=False)
    if candidate[slot1] != candidate[slot2]:
        dummy = candidate[slot1]
        candidate[slot1] = candidate[slot2]
        candidate[slot2] = dummy
    return candidate


"""def simulated_annealing_run(initial_temperature, final_temperature, a, initial_candidate, penalty_point,
                            num_consultant_rooms, len_parttime_shift, start_parttime_shift, alpha, beta, gamma):
    temperature = initial_temperature

    current_candidate = initial_candidate.copy()
    best_candidate = initial_candidate.copy()
    current_penalty_point = penalty_point
    best_penalty_point = penalty_point

    print("start SA: ", temperature)

    iteration = 0
    neighbourhood_structure_no = 2

    while temperature >= final_temperature:
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

        new_candidate = repair_chromosome(new_candidate, num_consultant_rooms, len_parttime_shift, start_parttime_shift)
        new_penalty_point = fitness_function(new_candidate, alpha, beta, gamma, len_parttime_shift, start_parttime_shift)
        difference = new_penalty_point - current_penalty_point

        if difference < 0 or np.random.random() < np.exp((-1 * difference) / temperature):
            current_candidate = new_candidate.copy()
            current_penalty_point = new_penalty_point.copy()

        if current_penalty_point < best_penalty_point:
            best_candidate = current_candidate.copy()
            best_penalty_point = current_penalty_point.copy()

        temperature *= a
        iteration += 1

        print("Iteration: ", iteration, " temperature: ", temperature, " best_candidate: ", best_candidate, " best_penalty_point: ", best_penalty_point)

    return best_candidate, best_penalty_point"""
