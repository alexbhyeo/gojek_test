import pandas as pd 
from datetime import datetime
data = pd.read_csv("/Users/yeoboonhong/Documents/gojek_assignment/ds-assignment-master/data/processed/transformed_dataset.csv")
print(data.head(10))
print(data.dtypes)

#print(data[["driver_gps_accuracy"]].groupby("driver_gps_accuracy").count())


def process_ranking_data(data):
    booking_data = data[data["participant_status"] != "CREATED"]
    print(booking_data.groupby("participant_status")["is_completed"].count())

    print(booking_data[booking_data["participant_status"] == "ACCEPTED"]["is_completed"].head(5))
    print(booking_data[booking_data["participant_status"] == "IGNORED"]["is_completed"].head(5))
    print(booking_data[booking_data["participant_status"] == "REJECTED"]["is_completed"].head(5))

    selected_data = booking_data[["trip_distance", "driver_distance", "event_hour", "driver_gps_accuracy", "is_completed"]]
    print(selected_data.head())

    return
process_ranking_data(data)
def process_history_data(data):
    proc_data = data.copy()
    proc_data['event_timestamp'] = pd.to_datetime(proc_data['event_timestamp'], format='mixed', utc=True)

    proc_data = proc_data.sort_values(['driver_id', 'event_timestamp'])

    unique_drivers = proc_data['driver_id'].unique()

    #print(proc_data.head(10))

    print(len(unique_drivers))

    print("proc_data length : ", len(proc_data))

    drivers_history = []
    for driver in unique_drivers:

        driver_data = proc_data[ (proc_data["driver_id"] == driver) & (proc_data['participant_status'] == 'ACCEPTED') ].copy()
        driver_data["booking_history"] = driver_data['is_completed'].cumsum()
        drivers_history.append(driver_data)

    leftover_data = proc_data[proc_data['participant_status'] != 'ACCEPTED']
    leftover_data["booking_history"] = 0
    drivers_history.append(leftover_data)


    result = pd.concat(drivers_history, ignore_index=True, sort=False)
    print("result length : ", len(result))

    print(result.head(100))
    return result
