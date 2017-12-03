import matplotlib.pyplot as plt
from matplotlib.dates import date2num, AutoDateFormatter, AutoDateLocator
import pandas as pd
import datetime

file_name = "../datasets/electricity_load/LD2011_2014_preprocessed.txt"
# skip header
dataset = pd.read_csv(file_name, low_memory=False, skiprows=[0])


# Data for plotting
time = dataset[dataset.columns[0:1]].values[::50].tolist()
values = dataset[dataset.columns[1:2]].values[::50].tolist()

# 2011-01-06 12:15:00 format
time = date2num([datetime.datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S") for x in time])

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()

ax.set(xlabel='time', ylabel='electricity load (after normalization)',
       title='Electricity load over 4 years period')


locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator))

ax.set_ylim([0, 1])

ax.grid()

ax.plot(time, values, 'o', markersize=2)

fig.autofmt_xdate()

fig.savefig("plotted.png")
