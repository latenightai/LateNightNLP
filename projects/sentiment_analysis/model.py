import torch 
import torch.nn as nn
import torch.nn.functional as F

class SentimentLSTM(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p=0.8):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first=True, dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_words, batch_size):
        embedded_words = self.embedding(input_words)
        lstm_out = self.dropout(lstm_out)
        lstm_out, h = lstm_out.contiguous().view(-1, self.n_hidden)
        fc_out = self.fc(lstm_out)
        sigmoid_out = self.sigmoid(fc_out)
        sigmoid_out = sigmoid_out.view(batch_size, -1)
        sigmoid_last = sigmoid_out[:, -1]
        return sigmoid_last, h

    def init_hidden(self, batch_size):
        device = 'gpu'
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
        weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device)
        )

        return h




