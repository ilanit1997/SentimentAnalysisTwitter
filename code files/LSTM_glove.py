from torch import nn
import torch

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim



class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, weights_matrix, output_size, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = False
        if torch.cuda.is_available():
            self.train_on_gpu = True


        # embedding and LSTM layers
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
        if n_layers > 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                dropout=drop_prob, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        # embeddings
        hidden = self.init_hidden(batch_size)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        soft_max_out = self.log_softmax(out)
        # # reshape to be batch_size first
        soft_max_out = soft_max_out.view(batch_size,  -1, self.output_size)
        soft_max_out = soft_max_out[:, -1]  # get last batch of labels

        return soft_max_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
