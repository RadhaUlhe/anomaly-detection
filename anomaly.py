import os
import pandas as pd
from adtk.data import validate_series
from adtk.detector import LevelShiftAD, InterQuartileRangeAD, ThresholdAD
from adtk.visualization import plot
from matplotlib import pyplot as plt
from adtk.pipe import Pipenet
from adtk.transformer import DoubleRollingAggregate
from adtk.aggregator import AndAggregator
import banpei


dirPath = "/home/dell/Desktop/anomaly detector/"
os.chdir(dirPath)

data = pd.read_csv(os.path.join("data2.csv"), index_col="timestamp", parse_dates=True, squeeze=True)
data_r = pd.read_csv(os.path.join("data2.csv"))
#data = pd.read_csv(os.path.join("data","artificialWithAnomaly","art_daily_jumpsup.csv"), index_col="timestamp", parse_dates=True, squeeze=True)

#data = pd.read_csv("Execution latency.csv", index_col="Time", parse_dates=True, squeeze=True)
#data = data.iloc[1:, [1]]

#data = data.sort_index()
s = validate_series(data)
#print(s)


# method 1
#level_shift_ad = LevelShiftAD(c=6.0, side='both', window=2)
#anomalies = level_shift_ad.fit_detect(s)

# method 2
s_transformed = DoubleRollingAggregate(
    agg="mean",
    window=10, #The tuple specifies the left window to be 3, and right window to be 1
    diff="l1").transform(s).rename("Diff rolling mean with different window size")

# method 3
iqr_ad = InterQuartileRangeAD(c=3)
anomalies_iqr_ad = iqr_ad.fit_detect(s_transformed)

#method 4
s_transformed_again = DoubleRollingAggregate(
    agg="mean",
    window=3, #The tuple specifies the left window to be 3, and right window to be 1
    diff="l1").transform(s).rename("Diff rolling mean with different window size")

#method 5
threshold_ad = ThresholdAD(high=0)
anomalies_threshold = threshold_ad.detect(s_transformed_again)


# method 3
steps = {
    "abs_level_change": {
        "model": DoubleRollingAggregate(
            agg="median",
            window=10,
            center=True,
            diff="l1"
        ),
        "input": "original"
    },
    "level_shift": {
        "model": InterQuartileRangeAD(c=3.0),
        "input": "abs_level_change"
    },
    "level_change": {
        "model": DoubleRollingAggregate(
            agg="median",
            window=10,
            center=True,
            diff="diff"
        ),
        "input": "original",
    },
    "positive_level_change": {
        "model": ThresholdAD(high=0),
        "input": "level_change"
    },
    "positive_level_shift": {
        "model": AndAggregator(),
        "input": ["level_shift", "positive_level_change"]
    }
}

pipenet = Pipenet(steps)
print(pipenet.get_params())

anomalies = pipenet.fit_detect(s)

# banpei
model = banpei.SST(w=30)
results = model.detect(data, is_lanczos=True)

# data_value = results.tolist()
# data_timestamp = data_r.loc[:, 'timestamp']
# d = {'value': data_value}
# r = pd.DataFrame(data=d)
# r_final = r.merge(data_r, right_index=True, left_index=True)
# r_final.set_index('timestamp', inplace=True)
# r_final = r_final.loc[:, 'value_x']
# anomalies_banpai = pipenet.fit_detect(r_final)

# plot
#data_plot = pd.read_csv("Execution latency.csv")
#data = data_plot.iloc[1:, [1, 2]]

fig, ax = plt.subplots(3)
ax[0].plot(anomalies, color="red")
ax[1].plot(s_transformed, color = "blue")
ax[1].plot(anomalies_iqr_ad, color="red")
ax[0].plot(s, color="blue")
ax[2].plot(results, color="black")
#ax[2].plot(anomalies_banpai, color="red")
#ax[2].plot(s_transformed_again, color="blue")
#ax[2].plot(anomalies_threshold, color="red")
plt.show()


#plot(s, anomaly=anomalies, ts_linewidth=1, anomaly_color='red')

print("success!")
