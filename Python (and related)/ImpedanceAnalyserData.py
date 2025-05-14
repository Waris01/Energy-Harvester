import matplotlib.pyplot as plt
import numpy as np

def read_impedance_data(filename):
    """Reads impedance data from Keysight CSV file"""
    frequencies = []
    impedance = []
    read_data = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('BEGIN CH1_DATA'):
                read_data = True
                next(f)  # Skip header line
                continue
                
            if read_data and line and not line.startswith('!'):
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        freq = float(parts[0])
                        z = float(parts[1])
                        frequencies.append(freq)
                        impedance.append(z)
                    except ValueError:
                        continue
                        
    return np.array(frequencies), np.array(impedance)

# Read data from both files
freq1, z1 = read_impedance_data('SPOEL1.CSV')
freq2, z2 = read_impedance_data('SPOEL2.CSV')

# Find peak values
peak_idx1 = np.argmax(z1)
peak_idx2 = np.argmax(z2)

peak_freq1, peak_z1 = freq1[peak_idx1], z1[peak_idx1]
peak_freq2, peak_z2 = freq2[peak_idx2], z2[peak_idx2]

# Create plot
plt.figure(figsize=(12, 6))
plt.semilogx(freq1, z1, label=f'SPOEL1 (Peak: {peak_z1:.2f} Ω @ {peak_freq1:.2f} Hz)')
plt.semilogx(freq2, z2, label=f'SPOEL2 (Peak: {peak_z2:.2f} Ω @ {peak_freq2:.2f} Hz)')

# Formatting
plt.title('Impedance vs Frequency', fontsize=14)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('|Z| (Ohm)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()

# Show plot
plt.show()
