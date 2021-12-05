import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from utils import sliding_windows, conv_and_reshape
import pandas as pd

from model import LSTM
from visualization import plot_results
from utils import conv_to_gif

# Computes ADE
def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.mean(axis=-1)                       # samples
        ade += dist.min(axis=0)                         # (1, )
    ade /= len(pred_arr)
    return ade

# Computes FDE
def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples 
        fde += dist.min(axis=0)                         # (1, )
    fde /= len(pred_arr)
    return fde

def compute_avg_ADE_FDE(lstm, test_df, test_inst_ids, frames_to_predict):
    results_look_ahead = {}
    avg_ade = 0
    avg_fde = 0
    for i in range(len(test_inst_ids)):
        predicted_data, data_expected, data_original = look_ahead_test(lstm, test_df, test_inst_ids, i, frames_to_predict, 0, seq_length)
        ade = compute_ADE(predicted_data, data_expected)
        fde = compute_FDE(predicted_data, data_expected)
        avg_ade +=ade
        avg_fde +=fde
        results_look_ahead[i] = [predicted_data, data_expected, data_original, ade, fde]

    avg_ade = avg_ade/len(test_inst_ids)
    avg_fde = avg_fde/len(test_inst_ids)
    return avg_ade, avg_fde

# Performs one step prediction
def one_step_test(lstm, test_df, test_inst_ids, test_id, frames_to_predict, seq_length):
        
    test_df = test_df[test_df.instance_id == test_inst_ids[test_id]]
    cols = list(test_df)[1:3]

    df_cols_testing = test_df[cols]

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_cols_testing = scaler.fit_transform(df_cols_testing)

    df_cols_testing_x, df_cols_testing_y = df_cols_testing[:,0], df_cols_testing[:,1]

    x, expected_x = sliding_windows(df_cols_testing_x, seq_length)
    y, expected_y = sliding_windows(df_cols_testing_y, seq_length)

    dataX, dataY = conv_and_reshape(x,y)
    expected_x, expected_y = conv_and_reshape(expected_x, expected_y)

    lstm.eval()
    with torch.no_grad():
        cat_data = torch.cat([dataX, dataY], dim=2)

        predicted_output_x, predicted_output_y = lstm(cat_data)
        combined_predicted_output = torch.cat([predicted_output_x, predicted_output_y], dim=1)
        combined_expected_traj = torch.cat([expected_x, expected_y], dim=1)

        data_predict = scaler.inverse_transform(combined_predicted_output)
        data_expected = scaler.inverse_transform(combined_expected_traj)

        # pred_data = np.reshape(predicted_data, (len(predicted_data),len(predicted_data[0])))

    return data_predict, data_expected
    # return combined_predicted_output, combined_expected_traj

# Performs look ahead prediction with only X and Y
def look_ahead_test(lstm, test_df, test_inst_ids, test_id, frames_to_predict, start, seq_length):
        
    test_df = test_df[test_df.instance_id == test_inst_ids[test_id]]
    cols = list(test_df)[1:3]

    df_cols_testing = test_df[cols]

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_cols_testing = scaler.fit_transform(df_cols_testing)

    df_cols_testing_x, df_cols_testing_y = df_cols_testing[:,0], df_cols_testing[:,1]

    x, expected_x = sliding_windows(df_cols_testing_x, seq_length)
    y, expected_y = sliding_windows(df_cols_testing_y, seq_length)

    x, expected_x = [x[start]], expected_x[start:]
    y, expected_y = [y[start]], expected_y[start:]

    original_x, original_y = df_cols_testing_x[:start+seq_length], df_cols_testing_y[:start+seq_length]

    dataX, dataY = conv_and_reshape(x,y)
    
    original_x, original_y = conv_and_reshape(original_x, original_y)
    expected_x, expected_y = conv_and_reshape(expected_x, expected_y)

    lstm.eval()
    with torch.no_grad():
        predicted_data = []
        cat_data = torch.cat([dataX, dataY], dim=2)

        for i in range(frames_to_predict):
            predicted_output_x, predicted_output_y = lstm(cat_data)

            combined_predicted_output = torch.cat([predicted_output_x, predicted_output_y], dim=1)

            new_cat_data = cat_data[0][1:]

            new_data = torch.cat([new_cat_data, combined_predicted_output])
            cat_data = torch.reshape(new_data, (1,new_data.shape[0], new_data.shape[1]))
            # print(f"round {i}:\n{cat_data}")

            data_predict = scaler.inverse_transform(combined_predicted_output)
            predicted_data.append(data_predict[0])

        combined_original_traj = torch.cat([original_x, original_y], dim=1)
        combined_expected_traj = torch.cat([expected_x, expected_y], dim=1)

        data_original = scaler.inverse_transform(combined_original_traj)
        data_expected = scaler.inverse_transform(combined_expected_traj)

        pred_data = np.reshape(predicted_data, (len(predicted_data),len(predicted_data[0])))

    return pred_data, data_expected, data_original

# Performs look ahead prediction with X,Y,Z and heading angle values
def look_ahead_test_all(lstm, test_df, test_inst_ids, test_id, frames_to_predict, start, seq_length):
        
    test_df = test_df[test_df.instance_id == test_inst_ids[test_id]]
    cols = list(test_df)[1:5]

    df_cols_testing = test_df[cols]

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_cols_testing = scaler.fit_transform(df_cols_testing)

    df_cols_testing_x, df_cols_testing_y = df_cols_testing[:,0], df_cols_testing[:,1]
    df_cols_testing_z, df_cols_testing_r = df_cols_testing[:,2],  df_cols_testing[:,3]

    x, expected_x = sliding_windows(df_cols_testing_x, seq_length)
    y, expected_y = sliding_windows(df_cols_testing_y, seq_length)
    z, expected_z = sliding_windows(df_cols_testing_z, seq_length)
    r, expected_r = sliding_windows(df_cols_testing_r, seq_length)

    x, expected_x = [x[start]], expected_x[start:]
    y, expected_y = [y[start]], expected_y[start:]
    z, expected_z = [z[start]], expected_z[start:]
    r, expected_r = [r[start]], expected_r[start:]

    original_x, original_y = df_cols_testing_x[:start+seq_length], df_cols_testing_y[:start+seq_length]
    original_z, original_r = df_cols_testing_z[:start+seq_length], df_cols_testing_r[:start+seq_length]

    dataX, dataY = conv_and_reshape(x,y)
    dataZ, dataR = conv_and_reshape(z,r)
    
    original_x, original_y = conv_and_reshape(original_x, original_y)
    original_z, original_r = conv_and_reshape(original_z, original_r)
    expected_x, expected_y = conv_and_reshape(expected_x, expected_y)
    expected_z, expected_r = conv_and_reshape(expected_z, expected_r)

    lstm.eval()
    with torch.no_grad():
        predicted_data = []
        cat_all_data = torch.cat([dataX, dataY, dataZ, dataR], dim=2)

        for i in range(frames_to_predict):
            predicted_output_x, predicted_output_y = lstm(cat_all_data)
            
            temp_z = torch.reshape(expected_z[i], (1,expected_z[i].shape[0]))
            temp_r = torch.reshape(expected_r[i], (1,expected_r[i].shape[0]))
            
            combined_predicted_output = torch.cat([predicted_output_x, predicted_output_y, temp_z, temp_r], dim=1)

            new_cat_data = cat_all_data[0][1:]

            new_data = torch.cat([new_cat_data, combined_predicted_output])
            cat_all_data = torch.reshape(new_data, (1,new_data.shape[0], new_data.shape[1]))
            # print(f"round {i}:\n{cat_all_data}")

            data_predict = scaler.inverse_transform(combined_predicted_output)

            predicted_data.append(data_predict[0][:2])

        combined_original_traj = torch.cat([original_x, original_y, original_z, original_r], dim=1)
        combined_expected_traj = torch.cat([expected_x, expected_y, expected_z, expected_r], dim=1)
        data_original = scaler.inverse_transform(combined_original_traj)
        data_expected = scaler.inverse_transform(combined_expected_traj)
        pred_data = np.reshape(predicted_data, (len(predicted_data),len(predicted_data[0])))

    return pred_data, data_expected[:,:2], data_original[:,:2]

if __name__ == "__main__":
    PATH = './checkpoints/checkpt_40.pt'
    input_size = 2
    hidden_size = 2
    num_layers = 1

    num_classes = 2
    seq_length = 4

    frames_to_predict = 6

    lstm = LSTM(input_size, hidden_size, num_layers)
    lstm.load_state_dict(torch.load(PATH))

    test_df = pd.read_csv('./csv/test.csv', parse_dates=['frame_id'], index_col='frame_id')
    test_inst_ids = np.unique(test_df["instance_id"].values)

    avg_ade, avg_fde = compute_avg_ADE_FDE(lstm, test_df, test_inst_ids, frames_to_predict)

    print(f"ADE: {avg_ade}")
    print(f"FDE: {avg_fde}")

    index = 19
    _, data_expected, _= look_ahead_test(lstm, test_df, test_inst_ids, index, frames_to_predict, 0, seq_length)
    num_of_times = len(data_expected)
    print(num_of_times)

    for i in range(num_of_times-5):
        print(f"Frame {i}")
        predicted_data, data_expected, data_original = look_ahead_test(lstm, test_df, test_inst_ids, index, frames_to_predict, i, seq_length)
        plot_results(i, predicted_data, data_expected, data_original)
    
    # conv_to_gif()

