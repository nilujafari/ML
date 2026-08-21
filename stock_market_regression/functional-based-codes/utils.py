import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_and_clean_data(file_path, drop_list):
    # print("Loading the data...")
    # read the data
    data = pd.read_csv(file_path)

    # drop redundant columns
    # print("Cleaning the data")
    selected_data = data.drop(drop_list, axis=1)

    # check the data
    # print(f"Number of null data: {selected_data.isnull().sum()}")
    # print(f"Number of duplicated data: {selected_data.duplicated().sum()}")


    # sort by date and make sure date items are datetiem
    selected_data["Date"] = pd.to_datetime(selected_data["Date"])
    # print("Load completed")

    return selected_data


def plot_data(data, title, x_axis, y_axis):
    plt.figure(figsize=(16, 8))
    plt.plot(data[x_axis], data[y_axis])
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid()
    plt.show()

def generate_windowed_data(data, input_days = 30, output_days = 7):
    X = []
    y = []
    print(f"Expected generated data length should be: {len(data) - input_days - output_days + 1}")

    counter = 0
    number_of_windows = len(data) - input_days - output_days + 1
    for index in range(number_of_windows):
        # print(f"index: {index} - len(data): {len(data)} - number of windows: {number_of_windows}")
        x_start_index = index
        x_end_index = index + input_days

        y_start_idx = x_end_index
        y_end_idx = x_end_index + output_days
        X.append(data[x_start_index : x_end_index])
        y.append(data[y_start_idx : y_end_idx]) 

        counter += 1
        if counter < 10:
            print(f"x_start_index: {x_start_index} - x_end_index: {x_end_index - 1} | y_start_index: {y_start_idx} - y_end_index {y_end_idx}")

    return X, y