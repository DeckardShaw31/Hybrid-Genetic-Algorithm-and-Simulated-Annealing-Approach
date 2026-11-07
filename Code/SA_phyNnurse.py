from penalty_phyNnurse import fitness_function
from GA_phyNnurse import check_validation
#from GA import repair_chromosome
import numpy as np
import random

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
MAX_ATTEMPTS = 1000   # added 2026-11-07: the original loops forever when no valid move exists


def neighbourhood_structure3(candidate, start_parttime_shift, len_parttime_shift):
    check = False
    attempts = 0
    while not check:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
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
        if attempts > MAX_ATTEMPTS:   # e.g. no part-time staff left to move back
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

"""def simulated_annealing_run(initial_temperature, initial_candidate, penalty_point, num_fulltime_shifts,
                            num_parttime_shifts, num_consultant_rooms, len_parttime_shift, start_parttime_shift,
                            alpha, beta, gamma):
    temperature = initial_temperature
    final_temperature = 0.01 * initial_temperature
    a = 0.99

    current_candidate = initial_candidate.copy()
    best_candidate = initial_candidate.copy()
    current_penalty_point = penalty_point
    best_penalty_point = penalty_point

    print("start SA: ", temperature)

    iteration = 0
    neighbourhood_structure_no = 6

    while temperature >= final_temperature:
        neighbourhood_structure = np.random.randint(neighbourhood_structure_no)
        new_candidate = current_candidate.copy()

        if neighbourhood_structure == 0:
            new_candidate = neighbourhood_structure1(new_candidate)
        elif neighbourhood_structure == 1:
            new_candidate = neighbourhood_structure2(new_candidate)
        elif neighbourhood_structure == 2:
            new_candidate = neighbourhood_structure3(new_candidate)
        elif neighbourhood_structure == 3:
            new_candidate = neighbourhood_structure4(new_candidate)
        elif neighbourhood_structure == 4:
            new_candidate = neighbourhood_structure5(new_candidate)
        else:
            new_candidate = neighbourhood_structure6(new_candidate)

        if check_validation(new_candidate, num_consultant_rooms, len_parttime_shift, start_parttime_shift) == True:
            new_penalty_point = fitness_function(new_candidate, alpha, beta, gamma, len_parttime_shift,
                                                 start_parttime_shift)
            print("new_candidate: ", new_candidate, " new_penalty_point: ", new_penalty_point)
            difference = new_penalty_point - current_penalty_point

            if difference < 0 or np.random.random() < np.exp((-1 * difference) / temperature):
                current_candidate = new_candidate.copy()
                current_penalty_point = new_penalty_point.copy()

            if current_penalty_point < best_penalty_point:
                best_candidate = current_candidate.copy()
                best_penalty_point = current_penalty_point.copy()

        temperature *= a
        iteration += 1

        print("Iteration: ", iteration, " temperature: ", temperature)
        print("best_candidate: ", best_candidate, " best_penalty_point: ", best_penalty_point, " current_candidate: ", current_candidate, " current_penalty_point: ", current_penalty_point)

    return best_candidate, best_penalty_point"""
