import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_and_calculate_for_steps(file_path, capacitance_nF):
    all_step_data = []
    current_step_info = None
    times_current_step = []
    voltages_current_step = []
    capacitance = capacitance_nF * 1e-9  # Convert nF to F

    step_info_regex = re.compile(r"Step Information: Vin=([^\s]+)\s*\(Run: (\d+/\d+)\)")
    data_line_regex_td = re.compile(r'^\s*(-?\d+\.?\d*(?:e[+\-]?\d+)?)\s+(-?\d+\.?\d*(?:e[+\-]?\d+)?)\s*$')

    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            line_stripped = line.strip()
            step_match = step_info_regex.search(line_stripped)

            if step_match:
                # Process the previous step's data if any
                if current_step_info and times_current_step and voltages_current_step:
                    process_step_data(current_step_info, times_current_step, voltages_current_step, capacitance, all_step_data)

                vin_str_from_re = step_match.group(1)
                run_info_str = step_match.group(2)
                current_step_info = {'Vin_str_raw': vin_str_from_re, 'run_info': run_info_str, 'Vin': None}

                val_str_cleaned = vin_str_from_re.lower()
                parsed_vin_value = None
                multiplier = 1.0

                if val_str_cleaned.endswith("mv"):
                    multiplier = 1e-3
                    val_str_cleaned = val_str_cleaned[:-2]
                elif val_str_cleaned.endswith("m"):
                    multiplier = 1e-3
                    val_str_cleaned = val_str_cleaned[:-1]
                elif val_str_cleaned.endswith("kv"):
                    multiplier = 1e3
                    val_str_cleaned = val_str_cleaned[:-2]
                elif val_str_cleaned.endswith("k"):
                    multiplier = 1e3
                    val_str_cleaned = val_str_cleaned[:-1]
                elif val_str_cleaned.endswith("v"):
                    val_str_cleaned = val_str_cleaned[:-1]

                try:
                    if val_str_cleaned:
                        parsed_vin_value = float(val_str_cleaned) * multiplier
                        current_step_info['Vin'] = parsed_vin_value
                except ValueError:
                    print(f"Warning (line {line_number}): Could not parse Vin from '{vin_str_from_re}'")

                times_current_step = []
                voltages_current_step = []

            elif current_step_info:
                data_match = data_line_regex_td.match(line_stripped)
                if data_match:
                    try:
                        time_val = float(data_match.group(1))
                        voltage_val = float(data_match.group(2))
                        times_current_step.append(time_val)
                        voltages_current_step.append(voltage_val)
                    except ValueError:
                        continue

        # Process last step after file end
        if current_step_info and times_current_step and voltages_current_step:
            process_step_data(current_step_info, times_current_step, voltages_current_step, capacitance, all_step_data)

    return all_step_data

def moving_average(data, window_size=3):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def process_step_data(step_info, times, voltages, capacitance, all_step_data_list):
    min_time_step = min(times) if times else 0
    max_time_step = max(times) if times else 0

    if len(times) < 2:
        step_data = {
            **step_info,
            'times': list(times), 'voltages': list(voltages), 'powers': [],
            'peak_power': 0, 'peak_power_time': 0, 'peak_power_voltage': 0,
            'min_time': min_time_step, 'max_time': max_time_step
        }
        all_step_data_list.append(step_data)
        return

    energies = [0.5 * capacitance * v**2 for v in voltages]

    # Calculate power (dE/dt)
    powers = []
    for i in range(1, len(energies)):
        dt = times[i] - times[i-1]
        if dt > 0:
            power_val = (energies[i] - energies[i-1]) / dt
            powers.append(power_val)
        else:
            powers.append(0)

    # Optional: smooth powers to reduce noise (uncomment if desired)
    # powers = moving_average(powers, window_size=5)
    # Adjust times and voltages arrays accordingly if smoothing used

    # Find peak positive power only
    positive_powers = [p if p > 0 else 0 for p in powers]
    if positive_powers:
        max_power_index = np.argmax(positive_powers)
        peak_power_val = positive_powers[max_power_index]
        peak_power_t = times[max_power_index + 1]  # offset by 1 because powers start from index 1
        peak_power_v = voltages[max_power_index + 1]
    else:
        peak_power_val = 0
        peak_power_t = 0
        peak_power_v = 0

    step_data = {
        **step_info,
        'times': list(times), 'voltages': list(voltages), 'powers': powers,
        'peak_power': peak_power_val, 'peak_power_time': peak_power_t, 'peak_power_voltage': peak_power_v,
        'min_time': min_time_step, 'max_time': max_time_step
    }
    all_step_data_list.append(step_data)

    # Debug print per step:
    print(f"Run {step_info['run_info']} | Vin = {step_info['Vin']:.6f} V | Peak Power = {peak_power_val:.3e} W")

    # Optional debug plot of power vs time per step (comment out if many steps)
    plt.figure(figsize=(6, 3))
    plt.plot(times[1:], powers, label='Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.title(f"Power vs Time for Vin={step_info['Vin']:.6f} V (Run {step_info['run_info']})")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_all_steps_results(all_step_data, file_name, capacitance_nF):
    plt.figure(figsize=(18, 10))
    num_steps_plotted = 0

    for step_data in all_step_data:
        if not step_data.get('times') or not step_data.get('voltages'):
            continue

        num_steps_plotted += 1
        label = f"Vin={step_data['Vin']:.3f}V | PkPwr={step_data['peak_power']:.2e}W"
        line_plot = plt.plot(step_data['times'], step_data['voltages'], label=label)
        color = line_plot[0].get_color()

        plt.scatter([step_data['peak_power_time']], [step_data['peak_power_voltage']],
                    color=color, edgecolor='black', s=80, zorder=5, label='_nolegend_')

        annotation = (f"PkPwr: {step_data['peak_power']:.2e} W\n"
                      f"t={step_data['peak_power_time']:.2e} s\n"
                      f"V={step_data['peak_power_voltage']:.2f} V")

        plt.annotate(annotation,
                     xy=(step_data['peak_power_time'], step_data['peak_power_voltage']),
                     xytext=(8, 8), textcoords='offset points', fontsize=7, color=color,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=color, lw=0.5))

    plt.title(f'Voltage vs Time (C = {capacitance_nF} nF)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Power vs Vin Plot (descending Vin)
    sorted_data = sorted([step for step in all_step_data if step.get('Vin') is not None], key=lambda x: x['Vin'])
    vins = [step['Vin'] for step in sorted_data]
    peak_powers = [step['peak_power'] for step in sorted_data]

    if vins and peak_powers:
        plt.figure(figsize=(10, 6))
        plt.plot(vins, peak_powers, 'o-', color='tab:red')
        plt.gca().invert_xaxis()  # Reverse x-axis for descending Vin
        plt.title('Peak Power vs Vin (Descending Vin)')
        plt.xlabel('Vin (V)')
        plt.ylabel('Peak Power (W)')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    CAPACITANCE_NF = 100.0
    input_file = 'Draft7.txt'

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
    else:
        all_step_data_results = parse_and_calculate_for_steps(input_file, CAPACITANCE_NF)

        if all_step_data_results:
            print(f"\nProcessed {len(all_step_data_results)} steps from {input_file}.")
            plot_all_steps_results(all_step_data_results, input_file, CAPACITANCE_NF)
        else:
            print("No valid steps processed.")
