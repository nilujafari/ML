import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import json

def read_and_clean_data(file_path, drop_list):
    print("Loading the data...")
    raw_data = pd.read_csv(file_path)

    selected_data = raw_data.drop(drop_list, axis=1)

    print(f"Data shape: {selected_data.shape}")
    print(f"Null values: {selected_data.isnull().sum()}")
    print(f"Duplicate rows: {selected_data.duplicated().sum()}")
    print("Load data completed.")
    
    print("Make sure date items are datetime")
    selected_data["Date"] =  pd.to_datetime(selected_data["Date"])

    return selected_data

def plot_data(data, title, x_axiz, y_axis):
    plt.figure(figsize=(16, 8))
    plt.plot(data[x_axiz], data[y_axis])
    plt.title(title)
    plt.xlabel(x_axiz)
    plt.ylabel(y_axis)
    plt.grid()
    plt.show()

def generate_sliding_windows(data, input_days, output_days, verbose=True):
    X = []
    y = []
    number_of_windows = len(data) - input_days - output_days + 1

    print(f"\n Expected generated data length should be: {number_of_windows}")

    counter = 0
    for index in range(number_of_windows):
        x_start_index = index
        x_end_index = index + input_days

        y_start_index = x_end_index
        y_end_index = x_end_index + output_days

        X.append(data[x_start_index : x_end_index])
        y.append(data[y_start_index : y_end_index])

        counter += 1
        if verbose and counter < 10:
            print(f"x_start_index: {x_start_index} - x_end_index: {x_end_index - 1} - y_start_index: {y_start_index} - y_end_index: {y_end_index}")
    X = np.array(X)
    y = np.array(y)
    return X, y


def scale_data(X_train, X_test, y_train, y_test):

    feature_scaler = MinMaxScaler()
    X_train_norm = feature_scaler.fit_transform(X_train)
    X_test_norm = feature_scaler.transform(X_test)

    target_scaler = MinMaxScaler()
    y_train_norm = target_scaler.fit_transform(y_train)
    y_test_norm = target_scaler.transform(y_test)

    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, target_scaler

print("Traning models ...")
def train_model(model, X_train_norm, y_train_norm):
    model.fit(X_train_norm, y_train_norm)

    return model

def predict_model(model, X_test_norm):
    prediction = model.predict(X_test_norm)

    return prediction

def inverse_transform_data(scaler, data):
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return scaler.inverse_transform(data)

def evaluate_models(y_test, prediction_models):
    results = []
    for key,value in prediction_models.items():
        mae = mean_absolute_error(y_test, value)
        mse = mean_squared_error(y_test, value)
        rmse = np.sqrt(mse)
        results.append({"Model":key, "MAE":mae, "MSE":mse, "RMSE":rmse})
    
    return pd.DataFrame(results).round(4)

def plot_model_evaluation(result_models, output_days):
    models = result_models["Model"]
    metric = ["MAE", "MSE", "RMSE"]
    colors = ["blue", "orange", "green"]

    model_positions = np.arange(len(models))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Comparison of Regression Models - {output_days}-Day Prediction")
    
    for item,metric in enumerate(metric):
        ax[item].bar(model_positions, result_models[metric], color=colors[item])
        ax[item].set_xticks(model_positions)
        ax[item].set_xticklabels(models, rotation=45)
        ax[item].set_xlabel("Models")
        ax[item].set_ylabel(metric)
        ax[item].grid(axis="y")
    plt.tight_layout()
    plt.show()

def save_results_json(results, file_path):
    plot_data = {"models": results["Model"].tolist(),
    "MAE": results["MAE"].tolist(),
    "MSE": results["MSE"].tolist(),
    "RMSE": results["RMSE"].tolist()}

    with open(file_path, "w") as f:
        json.dump(plot_data, f, indent=4)
    print("Saved Sucssesfull!")
