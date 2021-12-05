import torch
import torch.nn as nn
from utils import get_data, get_all_data
import pandas as pd
import numpy as np

from visualization import *
from model import LSTM


def train(lstm, train_df, train_inst_ids, seq_length, epoch, cols, optimizer, criterion):
    lstm.train()

    log_interval = 200
    train_loss = 0
    rep_loss = 0 

    for idx, inst_id in enumerate(train_inst_ids):
        # Loop around dataset
        optimizer.zero_grad()

        # forward propagation
        dataX, dataY, expected_x, expected_y = get_data(train_df, inst_id, seq_length, cols)
        # dataX, dataY, expected_x, expected_y, dataZ, dataR, expected_z, expected_r = get_all_data(train_df, inst_id, seq_length)
        
        cat_data = torch.cat([dataX, dataY], dim=2)
        # cat_all_data = torch.cat([dataX, dataY, dataZ, dataR], dim=2)

        predicted_output_x, predicted_output_y = lstm(cat_data)

        # obtain the loss function
        loss_x = criterion(predicted_output_x, expected_x)
        loss_y = criterion(predicted_output_y, expected_y)
        
        loss = loss_x + loss_y
        train_loss += loss
        rep_loss += loss
        
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0 and idx > 0:
            print(f"| Epoch {epoch:3d} | {idx:5d}/{len(train_inst_ids):5d} batches | loss: {rep_loss/log_interval:8.3f}")
            rep_loss = 0
    return train_loss/len(train_inst_ids)

def validation(lstm, val_df, val_inst_ids, seq_length, epoch, cols, criterion):
    lstm.eval()
    cum_loss = 0

    with torch.no_grad():
        for idx, inst_id in enumerate(val_inst_ids):
            # forward propagation
            dataX, dataY, expected_x, expected_y = get_data(val_df, inst_id, seq_length, cols)
            # dataX, dataY, expected_x, expected_y, dataZ, dataR, expected_z, expected_r = get_all_data(val_df, inst_id, seq_length)
            
            cat_data = torch.cat([dataX, dataY], dim=2)
            # cat_all_data = torch.cat([dataX, dataY, dataZ, dataR], dim=2)

            predicted_output_x, predicted_output_y = lstm(cat_data)

            # obtain the loss function
            loss_x = criterion(predicted_output_x, expected_x)
            loss_y = criterion(predicted_output_y, expected_y)

            loss = loss_x + loss_y
            cum_loss += loss
    print(f'| Validation Epoch {epoch:3d} | Avg-Val Loss: {cum_loss/(len(val_inst_ids))}') 

    return cum_loss/len(val_inst_ids)

if __name__ == "__main__":
    train_df = pd.read_csv('./csv/trg.csv', parse_dates=['frame_id'], index_col='frame_id')
    val_df = pd.read_csv('./csv/val.csv', parse_dates=['frame_id'], index_col='frame_id')
    test_df = pd.read_csv('./csv/test.csv', parse_dates=['frame_id'], index_col='frame_id')

    train_inst_ids =  np.unique(train_df["instance_id"].values)
    val_inst_ids = np.unique(val_df["instance_id"].values)
    test_inst_ids = np.unique(test_df["instance_id"].values)
    cols = list(train_df)[1:3]

    print(len(train_inst_ids))
    print(len(val_inst_ids))
    print(len(test_inst_ids))
    print(cols)

    num_epochs = 100
    learning_rate = 0.001

    input_size = 2
    hidden_size = 2
    num_layers = 1

    seq_length = 4

    tl = []
    vl = []

    lstm = LSTM(input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    min_val_loss = 1000
    stop_trigger = 0
    patience = 5

    # Train the model
    for epoch in range(1,num_epochs+1):
        train_loss = train(lstm, train_df, train_inst_ids, seq_length, epoch, cols, optimizer, criterion)
        val_loss = validation(lstm, val_df, val_inst_ids, seq_length, epoch, cols, criterion)
        tl.append(train_loss.item())
        vl.append(val_loss.item())

        if val_loss.item() <= round(min_val_loss,3):
            min_val_loss = val_loss.item()
            stop_trigger = 0 # Reset count
        
        elif val_loss.item() > round(min_val_loss,3):
            stop_trigger += 1
            print(f"| Trigger Count | {stop_trigger}/{patience}")

        if stop_trigger == patience:
            print("Early Stopped!!!")
            break

        torch.save(lstm.state_dict(), f'checkpoints/checkpt_{epoch}.pt')

    print("Min Val Loss: ", min_val_loss)

    plot_loss(tl, vl)