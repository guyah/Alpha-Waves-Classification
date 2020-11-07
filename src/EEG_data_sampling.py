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
#type_of_data = "Clinical_data"
type_of_data = "t_elec_15"
#type_of_data = "t_elec_13"
#folder_name = "MarRi0009"
folder_name = "MarRi_2_0008"
#folder_name = "clinical"
extension = ".vhdr"
sample_data_raw_file = os.path.join(parent_path, data_folder, type_of_data,
                                    str(folder_name)+str(extension))

print(sample_data_raw_file)



raw = mne.io.read_raw_brainvision(sample_data_raw_file,preload=True)

data = raw.filter(l_freq=8,h_freq=12,method="iir")
print(data.info)

raw_selection = data[0,:]


figure(num=None, figsize=(200, 50), dpi=90, facecolor='w', edgecolor='k')
x = raw_selection[1]
y = raw_selection[0].T



numbers = y.flatten()

window_size = 100


numbers_series = pd.Series(numbers)

windows = numbers_series.rolling(window_size)

moving_diff = windows.apply(find_avg_max_peaks)


moving_diff_list = moving_diff.tolist()

without_nans = moving_diff_list[window_size - 1:]

add = np.zeros(window_size-1)

without_nans = np.append(without_nans,add)


threshold = 40e-6

output = without_nans>threshold




#print(output)

zeros =  zero_runs(output)
fs = 2500

nbOfZeros = 7*fs
for i in zeros:
	if((i[1]-i[0]) < nbOfZeros):
		output[int(i[0]):int(i[1])] = 1 
		

zeros = zero_runs(~output)


df = pd.DataFrame({'x': x, 'y': y.flatten(), 'label':output})
output_name = str(folder_name)+str(".csv")

df.to_csv(os.path.join(parent_path, data_folder, type_of_data,output_name))

df.to_csv("output.csv")




# df = pd.read_csv("output.csv")
# print(df)
# output = df["label"].values.tolist()
# output = np.array(output)

# y = df["y"].values.tolist()
# y = np.array(y)

# x = df["x"].values.tolist()
# x = np.array(x)



# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Amplitude', color=color)
# ax1.set_ylim(-40e-6,40e-6)
# ax1.plot(x, y, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Label', color=color)  # we already handled the x-label with ax1
# ax2.plot(x, output, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.grid()
# plt.show()


