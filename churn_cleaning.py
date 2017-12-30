import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler


def cleaning_func(df):
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['churn'] = np.where(df['last_trip_date'] >= '2014-06-01', 0, 1)
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(4.777)
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(4.602)
    # df.fillna(df.mean(), inplace =True)
    df['phone'] = np.where(df['phone'] == 'Android', 'Android',
                           np.where(df['phone'] == 'iPhone', 'iPhone',
                           np.where(df['phone'] == 'test', 'test', 'Other')))
    df = pd.get_dummies(df, prefix=['city'], columns=['city'])
    df = pd.get_dummies(df, prefix=['phone'], columns=['phone'])
    df = pd.get_dummies(df, prefix=['luxury_car_user'],
                        columns=['luxury_car_user'], drop_first=True)
    df.drop(['last_trip_date'], inplace=True, axis=1)
    return df


def add_columns(cleaned_df):
    cleaned_df['days_since_signup'] = (pd.to_datetime('2014-07-01') -
                                       cleaned_df['signup_date']).dt.days
    cleaned_df['diff_ratings'] = (cleaned_df['avg_rating_of_driver'] -
                                  cleaned_df['avg_rating_by_driver'])
    cleaned_df.drop(['signup_date'], inplace=True, axis=1)
    return cleaned_df


def scale_data(cleaned_df):
    y_clean = cleaned_df.pop('churn')
    X_clean = cleaned_df

    X_col_names = X_clean.columns
    y_col_names = ['churn']

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean,
                                                        stratify=y_clean)
    scaler = StandardScaler()
    X_train_df = pd.DataFrame(X_train, columns=X_col_names)
    X_test_df = pd.DataFrame(X_test, columns=X_col_names)
    y_train_df = pd.DataFrame(y_train, columns=y_col_names)
    y_test_df = pd.DataFrame(y_test, columns=y_col_names)
    column_list = ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
                   'avg_surge', 'surge_pct', 'trips_in_first_30_days',
                   'weekday_pct', 'days_since_signup', 'diff_ratings']

    X_train_df[column_list] = scaler.fit_transform(X_train_df[column_list])
    X_test_df[column_list] = scaler.transform(X_test_df[column_list])
    return X_train_df, X_test_df, y_train_df, y_test_df, scaler


if __name__ == '__main__':
    df = pd.read_csv("data/churn_train.csv")
    cleaned_df = cleaning_func(df)
    cleaned_df = add_columns(cleaned_df)
    X_train, X_test, y_train, y_test, scaler = scale_data(cleaned_df)
