import simpy
import pandas as pd
import numpy as np
from ED_DES import run_emergency_dept

# import data
file_path = "Data.xlsx"
df_arrival_rates = pd.read_excel(file_path, sheet_name="Arrival rates")
arrival_rates = df_arrival_rates.iloc[:, 0].tolist()
df_service_times = pd.read_excel(file_path, sheet_name="Service times")
service_times = df_service_times.iloc[:, 0].tolist()

# initialize the environment and run the simulation
env = simpy.Environment()
stats = {}
hourly_stats = {}
all_values_stats = {}
patient_log = []
summary_stats = []

doctor_schedule = {
    0: 5,
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    5: 5,
    6: 5,
    7: 5,
    8: 5,
    9: 5,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
    14: 5,
    15: 5,
    16: 5,
    17: 5,
    18: 5,
    19: 5,
    20: 5,
    21: 5,
    22: 5,
    23: 5,
}

# set up
env.process(run_emergency_dept(env, stats=stats, num_triage_nurses=3, doctor_schedule=doctor_schedule, num_lab_tests=1,
                               num_ultrasonics=1,num_xrays=1, num_cts=1, warmup_days=10, total_days=1, patient_log=patient_log))
env.run()

# Export result to csv
for key, value in list(stats.items())[:24]:
    hourly = [sum(times) / len(times) if times else 0 for times in value]
    hourly_stats[key] = hourly

for key, value in list(stats.items())[:24]:
    all_values = [val for sublist in value for val in sublist]
    all_values_stats[key] = all_values

    if all_values:
        max_val = max(all_values)
        min_val = min(all_values)
        avg_val = sum(all_values) / len(all_values)
        std_val = np.std(all_values)
        count_val = len(all_values)
        total_val = sum(all_values)

        summary_stats.append({
            "Metric": key,
            "Max": max_val,
            "Min": min_val,
            "Avg": avg_val,
            "Std": std_val,
            "Count": count_val,
            "Total": total_val
        })

        print(f"{key}: Max={max_val:.2f}, Min={min_val:.2f}, Avg={avg_val:.2f}, Std={std_val:.2f}, Count={count_val:.2f}, Total={total_val:.2f}")
    else:
        summary_stats.append({
            "Metric": key,
            "Max": None,
            "Min": None,
            "Avg": None,
            "Std": None,
            "Total": 0
        })

        print(f"{key}: No data available")

print(sum(stats['total_wait_times']))
print(sum(stats['total_wait_times_with_examination']))
print(sum(stats['total_wait_times_without_examination']))

def save_hourly_stats(hourly_stats, filename="hourly_stats.csv"):
    df = pd.DataFrame(hourly_stats)
    df.to_csv(filename, index=False)

save_hourly_stats(hourly_stats)
def save_all_values_stats(all_values_stats, filename="all_values_stats.csv"):
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_values_stats.items()]))
    df.to_csv(filename, index=False)

save_all_values_stats(all_values_stats)

def save_summary_stats(summary_stats, filename="summary_stats.csv"):
    df = pd.DataFrame(summary_stats)
    df.to_csv(filename, index=False)

save_summary_stats(summary_stats)

def save_patient_log(patient_log, filename="patient_log.csv"):
    df = pd.DataFrame(patient_log)
    df.to_csv(filename, index=False)

save_patient_log(patient_log)

'''# Calculate and plot the average wait time for each hour for each queue
average_doctor_visit_wait_times = [sum(times) / len(times) if times else 0 for times in stats['hourly_wait_times_consultant_tp']]
average_queue_lengths_triage = [sum(lengths) / len(lengths) if lengths else 0 for lengths in stats['hourly_queue_lengths_triage']]
hourly_queue_lengths_doctors = [sum(lengths) / len(lengths) if lengths else 0 for lengths in stats['hourly_queue_lengths_consultant_tp']]
print(average_doctor_visit_wait_times)
# Plotting the average wait times for doctor visit
plt.figure(figsize=(10, 5))
plt.bar(range(24), average_doctor_visit_wait_times, color='green')
plt.title('Doctor visit Average Waiting Time Each Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Wait Time (minutes)')
plt.xticks(range(0, 24, 1))
plt.grid(True)
plt.show()

# Plotting the average queue lengths for triage
plt.figure(figsize=(10, 5))
plt.bar(range(24), average_queue_lengths_triage, color='blue')
plt.title('Average Triage Queue Length Each Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Queue Length')
plt.xticks(range(0, 24, 1))
plt.grid(True)
plt.show()

# Plotting the average queue lengths for Doctor visit
plt.figure(figsize=(10, 5))
plt.bar(range(24), hourly_queue_lengths_doctors, color='red')
plt.title('Average Doctor Visit Queue Length Each Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Queue Length')
plt.xticks(range(0, 24, 1))
plt.grid(True)
plt.show()

# Function to plot average queue lengths
def plot_queue_lengths(stats, title, color):
    average_lengths = [sum(lengths) / len(lengths) if lengths else 0 for lengths in stats]
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), average_lengths, color=color)
    plt.title(title)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Queue Length')
    plt.xticks(range(0, 24, 1))
    plt.grid(True)
    plt.show()

# Plot for each queue
plot_queue_lengths(stats['hourly_queue_lengths_lab_test'], 'Average Lab Test Queue Length Each Hour', 'purple')
plot_queue_lengths(stats['hourly_queue_lengths_ultrasonic'], 'Average Ultrasonic Queue Length Each Hour', 'orange')
plot_queue_lengths(stats['hourly_queue_lengths_xray'], 'Average XRay Queue Length Each Hour', 'yellow')
plot_queue_lengths(stats['hourly_queue_lengths_ct'], 'Average CT Queue Length Each Hour', 'green')

def plot_hourly_wait_times(stats, title, color):
    hourly_avg_wait_times = [sum(times) / len(times) if times else 0 for times in stats]
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), hourly_avg_wait_times, color=color)
    plt.title(title)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Wait Time (minutes)')
    plt.xticks(range(0, 24, 1))
    plt.grid(True)
    plt.show()

def plot_hourly_overall_waiting_time(stats):
    # Calculate average waiting times for each hour
    hourly_avg_waiting_times = [sum(times) / len(times) if times else 0 for times in stats['hourly_total_waiting_times']]
    plt.figure(figsize=(12, 6))
    plt.bar(range(24), hourly_avg_waiting_times, color='cyan')
    plt.title('Average Overall length of staying (LOS) per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Waiting Time (minutes)')
    plt.xticks(range(24))
    plt.grid(True)
    plt.show()

# Plot each queue's hourly waiting times for each different queue
plot_hourly_wait_times(stats['hourly_wait_times_lab_test'], 'Lab Test Hourly Average Wait Times', 'purple')
plot_hourly_wait_times(stats['hourly_wait_times_ultrasonic'], 'Ultrasonic Hourly Average Wait Times', 'blue')
plot_hourly_wait_times(stats['hourly_wait_times_xray'], 'XRay Hourly Average Wait Times', 'green')
plot_hourly_wait_times(stats['hourly_wait_times_ct'], 'CT Hourly Average Wait Times', 'red')
plot_hourly_overall_waiting_time(stats)


# Function to plot overall waiting time
def plot_overall_waiting_time(stats):
    total_times = stats['total_time_in_system']
    average_total_time = sum(total_times) / len(total_times) if total_times else 0
    plt.figure(figsize=(10, 5))
    plt.hist(total_times, bins=30, color='gray', alpha=0.7)
    plt.title('Distribution of Total Time in System (Triage to Leave EMD)')
    plt.xlabel('Total Time in System (minutes)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(average_total_time, color='red', linestyle='dashed', linewidth=1)
    plt.text(average_total_time + 5, plt.ylim()[1] * 0.9, f'Average: {average_total_time:.2f}', color='red')
    plt.show()

# Call the function to plot overall waiting time
plot_overall_waiting_time(stats)


def print_performance_stats(stats):
    # Calculate and print average and maximum wait times for various services
    for service in ['consultant_tp', 'consultant_ip', 'lab_test', 'ultrasonic', 'xray', 'ct']:
        wait_times = [time for sublist in stats[f'hourly_wait_times_{service}'] for time in sublist]
        if wait_times:  # Check if the list is not empty
            average_wait_time = sum(wait_times) / len(wait_times)
            max_wait_time = max(wait_times)
            print(f"Average Wait Time for {service.capitalize()} Service: {average_wait_time:.2f} minutes")
            print(f"Maximum Wait Time for {service.capitalize()} Service: {max_wait_time:.2f} minutes")
        else:
            print(f"No wait time data for {service.capitalize()} service.")
    # Print total patients processed
    total_patients = sum([len(times) for times in stats['hourly_total_waiting_times']])
    print(f"Total patients processed: {total_patients}")

fixed_capacities = {
    'triage': 1,  # 2 triage
    'consultant_tp': 2,  # 3 doctors
    'consultant_ip': 2,
    'lab_test': 3,  # 2 diagnostic
    'ultrasonic': 3,  # 4 treatment
    'xray': 2,  # 1 discharge
    'ct': 1   # 1 admission
}
def print_performance2_stats(stats, fixed_capacities):
    total_simulation_minutes = 24 * 60 * 30  # Total minutes in the simulation
    for service, capacity in fixed_capacities.items():
        # Calculate total service time for the service
        service_time = sum([sum(times) for times in stats[f'hourly_service_times_{service}']])
        # Calculate total available time for the service
        total_capacity_time = capacity * total_simulation_minutes
        # Calculate utilization rate
        utilization_rate = (service_time / total_capacity_time) * 100 if total_capacity_time > 0 else 0
        print(f"Utilization Rate for {service.capitalize()}: {utilization_rate:.2f}%")
    # Print total patients processed
    total_patients = sum([len(times) for times in stats['hourly_total_waiting_times']])
    print(f"Total patients processed: {total_patients}")

print_performance_stats(stats)
print_performance2_stats(stats, fixed_capacities)'''
