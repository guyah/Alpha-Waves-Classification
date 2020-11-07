import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from mne.preprocessing import peak_finder

def find_avg_max_peaks(y):
	_, max_peak_mags = peak_finder(y,extrema=1,verbose=False) 
	avg_max = np.mean(max_peak_mags)
	_, min_peak_mags = peak_finder(y,extrema=-1,verbose=False) 
	avg_min = np.mean(min_peak_mags)
	return np.abs(avg_max - avg_min)

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges



parent_path = ".."
data_folder = "data"
type_of_data = "Clinical_data"
#type_of_data = "t_elec_15"
#type_of_data = "t_elec_13"
#folder_name = "MarRi0001"
#folder_name = "MarRi_2_0007"
folder_name = "clinical"
extension = ".csv"

df = pd.read_csv(os.path.join(parent_path, data_folder, type_of_data,str(folder_name)+str(extension)))

#print(df)

output = df["label"].values.tolist()
output = np.array(output)

y = df["y"].values.tolist()
y = np.array(y)

x = df["x"].values.tolist()
x = np.array(x)



fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Amplitude', color=color)
ax1.set_ylim(-40e-6,40e-6)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Label', color=color)  # we already handled the x-label with ax1
ax2.plot(x, output, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()
