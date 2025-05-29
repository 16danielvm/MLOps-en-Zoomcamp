#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

from prefect import flow, task

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task(retries=3, retry_delay_seconds=2,
      name="Read DataFrame Task", 
      tags=["data_reading", "parquet"])
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

@task(retries=3, retry_delay_seconds=2,
      name="Create Feature Matrix Task", 
      tags=["feature_extraction", "dict_vectorization"])
def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task(
    retries=3,
    retry_delay_seconds=2,
    name="Train Model Task",
    tags=["model_training", "xgboost", "mlflow"],
)
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run(run_name='taxi_pred_xgboost_orchestration5') as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

@flow
def run(train_year, train_month, val_year, val_month):
    df_train = read_dataframe(year=train_year, month=train_month)
    df_val = read_dataframe(year=val_year, month=val_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

@flow
def master_flow(train_date: str = None, val_date: str = None):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    if train_date is None or val_date is None:
        today = datetime.today()
        train_date = today - relativedelta(months=3)
        val_date = today - relativedelta(months=2)
    else:
        train_date = datetime.strptime(train_date, "%Y-%m-%d")
        val_date = datetime.strptime(val_date, "%Y-%m-%d")

    run_id = run(
        train_year=train_date.year, train_month=train_date.month,
        val_year=val_date.year, val_month=val_date.month
    )
    print(f"Run ID: {run_id}")




if __name__ == "__main__":
    import argparse
    from prefect.schedules import Schedule

    parser = argparse.ArgumentParser(description='Orquestación para predicción de duración de taxis.')
    parser.add_argument('--train_date', type=str, help='Fecha de entrenamiento (formato YYYY-MM-DD)')
    parser.add_argument('--val_date', type=str, help='Fecha de validación (formato YYYY-MM-DD)')
    
    args = parser.parse_args()

    if args.train_date and args.val_date:
        # Caso 1: Backfill manual
        run_id = master_flow(train_date=args.train_date, val_date=args.val_date)
    else:
        # Caso 2: Programación automática con schedule
        master_flow.serve(
            name="duration-prediction-orchestration",
            tags=["duration-prediction", "orchestration"],
            description="Orchestration flow for duration prediction using Prefect",
            version="2.0.0",
            parameters={},  # Usa la lógica interna de fechas por defecto
            schedule=Schedule(
                cron="0 1 1 * *",  # 1ro de cada mes a las 9:00 AM
                timezone="America/Mexico_City"
            )
        )

# prefect server start antes de ejecutar el flow
# En otro terminal: python duration-prediction-orchestration_module.py --train_date 2023-01-01 --val_date 2023-02-01