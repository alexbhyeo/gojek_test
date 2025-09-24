import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")
    dataset = apply_feature_engineering(dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    #engineered_features = df.pipe(driver_distance_to_pickup).pipe(hour_of_day)    
    #return engineered_features

    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(driver_historical_completed_bookings)
    )

def apply_feature_engineering_on_test(df: pd.DataFrame) -> pd.DataFrame:
    #engineered_features = df.pipe(driver_distance_to_pickup).pipe(hour_of_day)    
    #return engineered_features

    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day) # driver_historical_completed_bookings is not used 
    )

    

if __name__ == "__main__":
    main()
