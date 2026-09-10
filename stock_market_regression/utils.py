import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import json
from enum import Enum

# MODEL NAMES
class AcceptableModelNames(Enum):

    LR = "LR"
    KNN = "KNN"
    DT = "DT"
    RF = "RF"
    MLP = "MLP"

# READ AND CLEAN DATA
def read_and_clean_data(file_path, drop_list):

    """
    Load the dataset, remove unwanted columns, check data quality,
    and convert the Date column to datetime.
    """
    print("Loading the data...")

    raw_data = pd.read_csv(file_path)

   # Check requested columns
    missing_columns = [col for col in drop_list if col not in raw_data.columns]

    if missing_columns :
        raise ValueError(f"The following columns were not found in the dataset:"f"{missing_columns}")

    # Drop unwanted columns
    data = raw_data.drop(drop_list, axis=1)

    print(f"Data shape: {data.shape}")
    print(f"Null values: {data.isnull().sum()}")
    print(f"Duplicate rows: {data.duplicated().sum()}")

    # Date column
    if "Date" not in data.columns:
        raise ValueError("The dataset must contain a 'Date' column.")
    
    print("Converting Date column to datetime...")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

     # Check invalid dates
    if data["Date"].isnull().any():
        raise ValueError("Some values in Date column could not be converted.")

    # Sort chronologically
    data = data.sort_values("Date").reset_index(drop=True)

    print("Load data completed.")

    return data

# Plot data
def plot_data(data, title, x_axis, y_axis):
    plt.figure(figsize=(16, 8))
    plt.plot(data[x_axis], data[y_axis])
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid()
    plt.tight_layout()
    plt.show()

# Generate Sliding Windows
def generate_sliding_windows(data, input_days,output_days, verbose=False):
    """
    Generate supervised learning windows.
    X:Previous input_days
    y:Next output_days
    Example:
        Input = 30 days
        Output = 7 days
        X = Days 1-30
        y = Days 31-37
        """
    
    X = []
    y = []

    number_of_windows = len(data) - input_days - output_days + 1
    if number_of_windows <= 0:
        raise ValueError("Not enough data to create windows.")
    
    print(f"Input : {input_days} days | "
        f"Output : {output_days} days | "
        f"Windows: {number_of_windows}")

    counter = 0
    for index in range(number_of_windows):
        x_start_index = index
        x_end_index = index + input_days

        y_start_index = x_end_index
        y_end_index = x_end_index + output_days

        X.append(data[x_start_index : x_end_index])
        y.append(data[y_start_index : y_end_index])

        counter += 1
        if verbose and counter <= 10:
            print(f"x_start_index: {x_start_index} - x_end_index: {x_end_index - 1} - y_start_index: {y_start_index} - y_end_index: {y_end_index}")
    X = np.array(X)
    y = np.array(y)

    return X, y

# def flatten_X(X):
#     """
#     Convert: (samples, input_days, features)
#     into: (samples, input_days * features)
#     """

#     return X.reshape(X.shape[0],-1)

# Train / Test Split
def split_data(window_data, output_days, test_size, shuffle):

    train_test_data = {}

    for day in output_days:
        X = window_data[day]["X"]
        y = window_data[day]["y"]

        X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size= test_size,
                shuffle= shuffle
            )
        
        train_test_data[day] = {
                "X_train" : X_train,
                "X_test" : X_test,
                "y_train" : y_train,
                "y_test" : y_test
                }
        
    return train_test_data

# Scale Data
def scale_data(X_train, X_test, y_train, y_test):

    feature_scaler = MinMaxScaler()

    X_train_norm = feature_scaler.fit_transform(X_train)
    X_test_norm = feature_scaler.transform(X_test)

    target_scaler = MinMaxScaler()

    y_train_norm = target_scaler.fit_transform(y_train)
    y_test_norm = target_scaler.transform(y_test)

    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, target_scaler

# Scale all data
def scale_all_data(train_test_data):

    scaled_data = {}

    for day in train_test_data:
        X_train_norm, X_test_norm, y_train_norm, y_test_norm, target_scaler = scale_data(
            train_test_data[day]["X_train"],
            train_test_data[day]["X_test"],
            train_test_data[day]["y_train"],
            train_test_data[day]["y_test"]
        )

        scaled_data[day] = {
            "X_train_norm" : X_train_norm,
            "X_test_norm" : X_test_norm,
            "y_train_norm" : y_train_norm,
            "y_test_norm" : y_test_norm,
            "target_scaler" : target_scaler
            }
        
    return scaled_data

# Train Regression Models
def train_models(model_name: AcceptableModelNames, X_train_norm, y_train_norm):

    if model_name == AcceptableModelNames.LR:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    elif model_name == AcceptableModelNames.KNN:
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors= 5)

    elif model_name == AcceptableModelNames.DT:
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state= 40)

    elif model_name == AcceptableModelNames.RF:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    elif model_name == AcceptableModelNames.MLP:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes= (100,), max_iter= 500, random_state= 40)
    else:
        print(f"NO MODEL FOUND FOR: {model_name}")
        raise Exception("The defined model doesn't exist in the acceptable list which includes: [LR, KNN, DT, RF, MLP]")

    # Flatten y if only one output
    if y_train_norm.shape[1] == 1:
        y_train_norm = y_train_norm.ravel()

    model.fit(X_train_norm, y_train_norm)

    return model

# Train all models
def train_all_models(models_list, output_days, scaled_data):
    trained_models = {}

    for model in models_list:
        trained_models[model.value] = {}

        print(f"\nTraining {model.value}...")

        for day in output_days:
            X_train = scaled_data[day]["X_train_norm"]
            y_train = scaled_data[day]["y_train_norm"]

            trained_models[model.value][day] = train_models(model, X_train, y_train)

        print(f"{model.value} training completed.")

    return trained_models

# Predictions
def predict_model(model, X_test_norm):

    return model.predict(X_test_norm)

# Predictions all models
def predict_all_models(models_list, trained_models, output_days, scaled_data):

    predictions = {}

    for model in models_list:
        predictions[model.value] = {}

        for day in output_days:
            model_object = trained_models[model.value][day]
            X_test = scaled_data[day]["X_test_norm"]
            predictions[model.value][day] = predict_model(model_object,X_test)

    return predictions

# inverse transform
def inverse_transform_data(scaler, data):
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return scaler.inverse_transform(data)

# Inverse transform predictions
def inverse_transform_all_predictions(models_list, predictions, output_days, scaled_data):

    predictions_real = {}

    for model in models_list:
        predictions_real[model.value] = {}

        for day in output_days:
            target_scaler = scaled_data[day]["target_scaler"]

            predictions_real[model.value][day] = inverse_transform_data(
                target_scaler,
                predictions[model.value][day]
                )

    return predictions_real

# Inverse transform test target
def inverse_transform_y_test(output_days, scaled_data):

    y_test_real = {}

    for day in output_days:
        target_scaler = scaled_data[day]["target_scaler"]

        y_test_real[day] = inverse_transform_data(
            target_scaler,
            scaled_data[day]["y_test_norm"]
        )

    return y_test_real

# Evaluation
def evaluate_models(y_test, prediction_models):

    evalu_result = []

    y_test = np.asarray(y_test)

    for model_name,prediction  in prediction_models.items():

        prediction = np.asarray(prediction)

        mae = mean_absolute_error(y_test, prediction )
        mse = mean_squared_error(y_test, prediction )
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, prediction )

        evalu_result.append({"Model":model_name, "MAE":mae, "MSE":mse, "RMSE":rmse, "R2":r2})
    
    return pd.DataFrame(evalu_result).round(4)

# Evaluation all models
def evaluate_all_models(predictions_real, output_days, y_test_real):

    evaluate_results = {}

    for day in output_days:
        predictions_models = {}

        for model_name, predictions in predictions_real.items():
            predictions_models[model_name] = predictions[day]

        evaluate_results[day] = evaluate_models(y_test_real[day], predictions_models)

    return evaluate_results

# plot models evaluation
def plot_model_evaluation(result_models, output_days):

    models = result_models["Model"]
    metrics = ["MAE", "MSE", "RMSE", "R2"]
    colors = ["red", "orange", "blue", "green"]

    model_positions = np.arange(len(models))

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle(f"Comparison of Regression Models - {output_days}-Day Prediction")

    for index, metric in enumerate(metrics):
        axes[index].bar(model_positions,result_models[metric],color=colors[index])
        axes[index].set_xticks(model_positions)
        axes[index].set_xticklabels(models, rotation=45)
        axes[index].set_xlabel("Models")
        axes[index].set_ylabel(metric)
        axes[index].set_title(metric)
        axes[index].grid(axis="y")

    plt.tight_layout()
    plt.show()

# Save results to json
def save_results_json(results, file_path):

    plot_data = {"models": results["Model"].tolist(),
    "MAE": results["MAE"].tolist(),
    "MSE": results["MSE"].tolist(),
    "RMSE": results["RMSE"].tolist(),
    "R2": results["R2"].tolist()}

    with open(file_path, "w") as file:
        json.dump(plot_data, file, indent=4)
    print(f"Results saved successfully: {file_path}")
