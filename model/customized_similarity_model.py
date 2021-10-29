import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
        att_w : size (N, T, 1)

        return:
        utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_logits = self.W(batch_rep).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep


class SpeakerLevelModel(nn.Module):
    def __init__(self, input_dim):
        super(SpeakerLevelModel, self).__init__()
        self.pooling = SelfAttentionPooling(input_dim)
        self.linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, features_x, features_y):
        """
        input:
        features_x : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        features_y : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
        att_w : size (N, T, 1)

        return:
        sim : scalar
        """
        device = features_x[0].device
        lengths_x = torch.LongTensor([len(feature) for feature in features_x]).to(
            device
        )
        features_x_padding_mask = ~torch.lt(
            torch.arange(max(lengths_x)).unsqueeze(0).to(device),
            lengths_x.unsqueeze(1),
        )
        padded_features_x = pad_sequence(features_x, batch_first=True)

        x = self.pooling(padded_features_x, features_x_padding_mask)
        x = self.linear(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)

        lengths_y = torch.LongTensor([len(feature) for feature in features_y]).to(
            device
        )
        features_y_padding_mask = ~torch.lt(
            torch.arange(max(lengths_y)).unsqueeze(0).to(device),
            lengths_y.unsqueeze(1),
        )
        padded_features_y = pad_sequence(features_y, batch_first=True)

        y = self.pooling(padded_features_y, features_y_padding_mask)
        y = self.linear(y)
        y = self.dropout(y)
        y = y.unsqueeze(2)

        sim = torch.matmul(x, y).squeeze().view(-1)

        return sim


class UtteranceLevelModel(nn.Module):
    def __init__(self, input_dim):
        super(UtteranceLevelModel, self).__init__()
        self.pooling = SelfAttentionPooling(input_dim)
        self.linear = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, features):
        """
        input:
        features : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
        att_w : size (N, T, 1)

        return:
        sim : scalar
        """
        device = features[0].device
        lengths = torch.LongTensor([len(feature) for feature in features]).to(device)
        features_padding_mask = ~torch.lt(
            torch.arange(max(lengths)).unsqueeze(0).to(device), lengths.unsqueeze(1),
        )
        padded_features = pad_sequence(features, batch_first=True)

        feature = self.pooling(padded_features, features_padding_mask)
        feature = self.linear(feature)
        feature = self.relu(self.dropout(feature))
        sim = self.output(feature)
        sim = sim.squeeze().view(-1)

        return sim
