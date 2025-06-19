import pickle
import pandas as pd

import sys
#import os

categorical = ['PULocationID', 'DOLocationID']

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def apply_model(year, month):

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    dv, model = load_model()

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    #y_pred.std()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['pred'] = y_pred

    #output_file = f'predictions_{year:04d}_{month:02d}.parquet'

    #df_result.to_parquet(
    #    output_file,
    #    engine='pyarrow',
    #    compression=None,
    #    index=False
    #)

    #file_size_bytes = os.path.getsize(output_file)
    #file_size_mb = file_size_bytes / (1024 * 1024)

    #print(f"File size: {file_size_mb:.2f} MB")

    print(f"Mean predicted duration: {df_result['pred'].mean():.2f}")

if __name__ == "__main__":
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    apply_model(year, month)
 
