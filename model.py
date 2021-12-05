import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Tensors in (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, 1)

        # self.hidden1 = nn.Linear(hidden_size, hidden_size)
        # self.hidden2 = nn.Linear(hidden_size, hidden_size)
        # self.hidden3 = nn.Linear(hidden_size, hidden_size)
        # self.hidden4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):
        h_0 = Variable(torch.zeros(
            self.num_layers, data.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, data.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        out, (_, _) = self.lstm(data, (h_0, c_0))

        out = out[:,-1,:]
        
        # out = self.hidden4(self.hidden3(self.hidden2(self.hidden1(out))))
        
        out_x = self.fc1(out)
        out_y = self.fc2(out)

        return out_x, out_y