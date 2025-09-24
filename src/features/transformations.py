import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    print(df.head(10))
    print(df.dtypes)
    return df

def remove_create_data(df: pd.DataFrame) -> pd.DataFrame:
    proc_data = df.copy()
    proc_data = proc_data[proc_data['participant_status'] != 'ACCEPTED']
    print(proc_data.groupby("participant_status")["is_completed"].count())
    print(proc_data[proc_data["participant_status"] == "ACCEPTED"]["is_completed"].head(5))
    print(proc_data[proc_data["participant_status"] == "IGNORED"]["is_completed"].head(5))
    print(proc_data[proc_data["participant_status"] == "REJECTED"]["is_completed"].head(5))

    
    return proc_data
    

def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    # sort the df according to event timestamp
    # select participant_status = accepted
    # create to show 
    proc_data = df.copy()

    proc_data = proc_data[proc_data["participant_status"] != "CREATED"]
    
    proc_data['event_timestamp'] = pd.to_datetime(proc_data['event_timestamp'], format='mixed', utc=True)

    proc_data = proc_data.sort_values(['driver_id', 'event_timestamp'])
    unique_drivers = proc_data['driver_id'].unique()

    print(len(unique_drivers))

    drivers_history = []
    for driver in unique_drivers:

        driver_data = proc_data[ (proc_data["driver_id"] == driver) & (proc_data['participant_status'] == 'ACCEPTED') ].copy()
        driver_data["booking_history"] = driver_data['is_completed'].cumsum()
        drivers_history.append(driver_data)

    leftover_data = proc_data[proc_data['participant_status'] != 'ACCEPTED']
    leftover_data["booking_history"] = 0
    drivers_history.append(leftover_data)

    result = pd.concat(drivers_history, ignore_index=True, sort=False)
    print(result.groupby("participant_status")["is_completed"].count())
    print(result[result["participant_status"] == "ACCEPTED"]["is_completed"].head(5))
    print(result[result["participant_status"] == "IGNORED"]["is_completed"].head(5))
    print(result[result["participant_status"] == "REJECTED"]["is_completed"].head(5))

    return result
    
