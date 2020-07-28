
import numpy as np
import scipy as sp
import os
import logging
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import random_split

from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')


def initialize_weights(model):
    print('initialize weights')
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight.data)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_uniform_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)


class LSTM_2L(nn.Module):
    def __init__(self, n_features = 1, hidden_dims = [80,80], seq_length = 250,
                 batch_size = 64, n_predictions = 10, dropout = 0.3):
        super(LSTM_2L, self).__init__()
        print ('LSTM_2L', n_features, hidden_dims, seq_length, batch_size, n_predictions, dropout)

        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.seq_length = seq_length
        self.num_layers = len(self.hidden_dims)
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.hidden_dims[0],
            batch_first = True,
            dropout = dropout,
            num_layers = 2)

        self.linear = nn.Linear(self.hidden_dims[1], n_predictions)
        self.init_hidden_state()

    def init_hidden_state(self):
        #initialize hidden states (h_n, c_n)

        print('init hidden state')

        # hidden[0] size -> (2, batch_size, hidden dim0)
        # hidden[1] size -> (2, hidden dim0, hidden_dim1)
        print ('Hidden dimension 0: ', self.num_layers, self.batch_size, self.hidden_dims[0])
        print ('Hidden dimension 1: ', self.num_layers, self.batch_size, self.hidden_dims[0])
        #print ('Hidden dimension 1: ', self.num_layers, self.hidden_dims[0], self.hidden_dims[1])
        self.hidden = (
            torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0]), #.to(self.device),
            #torch.randn(self.num_layers, self.hidden_dims[0], self.hidden_dims[1]) #.to(self.device)
            torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0]), #.to(self.device),
            )

    def forward(self, sequences):

        try:
            batch_size, seq_len, n_features = sequences.size()  # batch first
            print ('forward: ', batch_size, seq_len, n_features)
        except Exception:
            print ('forward issue', sequences)

        #hidden[0] = h_n, hidden[1] = c_n
        lstm1_out , (h1_n, c1_n) = self.lstm1(sequences, (self.hidden[0], self.hidden[1]))

        last_time_step = lstm1_out[:,-1,:]

        y_pred = self.linear(last_time_step)

        return y_pred


class Model:
    def __init__(self, config, run_id, channel, Path=None, Train=True):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.name = "Model"
        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.history = None

        if Path is None:
            Path = ""

        # bypass default training in constructor
        if not Train:
            self.new_model((None, channel.X_train.shape[2]))
        elif not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                path = os.path.join(Path, 'data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))
                self.train_new(channel)
                self.save(Path)
        else:
            self.train_new(channel)
            self.save(Path)

    def __str__(self):
        out = '\n%s:%s' % (self.__class__.__name__, self.name) + "\n" + str(self.model.summary())
        return out

    def load(self):
        """
        Load model for channel.
        """

        logger.info('Loading pre-trained model')
        self.model = load_model(os.path.join('data', self.config.use_id,
                                             'models', self.chan_id + '.h5'))

    def new_model(self, Input_shape):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        if self.model is not None:
            return

        self.model = LSTM_2L(n_features = Input_shape[1], hidden_dims = self.config.layers,
                 seq_length = self.config.l_s, batch_size = self.config.lstm_batch_size,
                 n_predictions = self.config.n_predictions, dropout = self.config.dropout)

        print ('input shape: ', Input_shape)

        return
        self.model = Sequential()

        self.model.add(LSTM(
            self.config.layers[0],
            input_shape=Input_shape,
            return_sequences=True))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(LSTM(
            self.config.layers[1],
            return_sequences=False))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(Dense(
            self.config.n_predictions))
        self.model.add(Activation('linear'))

        self.model.compile(loss=self.config.loss_metric,
                           optimizer=self.config.optimizer)

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        # instatiate model with input shape from training data
        self.new_model((None, channel.X_train.shape[2]))

        self.model.apply(initialize_weights)

        training_losses = []
        validation_losses = []

        loss_function = nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters())

        train_hist = np.zeros(self.config.epochs)

        X_train, X_validation, y_train, y_validation = train_test_split(
            channel.X_train, channel.y_train, train_size=0.8)

        print ('Shapes: ', channel.X_train.shape, channel.y_train.shape)
        print ('Training shapes: ', X_train.shape, y_train.shape)
        print ('Validation shapes: ', X_validation.shape, y_validation.shape)
        train_dataset=TensorDataset(torch.Tensor(X_train),torch.Tensor(y_train))
        validation_dataset=TensorDataset(torch.Tensor(X_validation),torch.Tensor(y_validation))

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config.lstm_batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(dataset=validation_dataset, batch_size=self.config.lstm_batch_size, drop_last=True, shuffle=True)

        self.model.train()

        print("Beginning model training...")

        for t in range(self.config.epochs):
            train_losses_batch = []
            print ('Epoch ', t)

            i = 0
            for X_batch_train, y_batch_train in train_loader:
                print ('Batch ', i)
                i += 1
                y_hat_train = self.model(X_batch_train)
                loss = loss_function(y_hat_train.float(), y_batch_train)
                train_loss_batch = loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses_batch.append(train_loss_batch)

            training_loss = np.mean(train_losses_batch)
            print ('After batch ', i-1, training_loss)
            training_losses.append(training_loss)

            with torch.no_grad():
                val_losses_batch = []
                for X_val_batch, y_val_batch in val_loader:
                    self.model.eval()
                    y_hat_val = self.model(X_val_batch)
                    val_loss_batch = loss_function(y_hat_val.float(), y_val_batch).item()
                    val_losses_batch.append(val_loss_batch)
                validation_loss = np.mean(val_losses_batch)
                validation_losses.append(validation_loss)

            print(f"[{t+1}] Training loss: {training_loss} \t Validation loss: {validation_loss} ")
            if training_loss < 0.02 and validation_loss < 0.02:
                break

        print('Training complete...')

        return self.model.eval()


        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=0)]

        self.history = self.model.fit(channel.X_train,
                                      channel.y_train,
                                      batch_size=self.config.lstm_batch_size,
                                      epochs=self.config.epochs,
                                      validation_split=self.config.validation_split,
                                      callbacks=cbs,
                                      verbose=True)

    def save(self, Path=None):
        """
        Save trained model.
        """
        if Path is None:
            Path = ""

        self.model.save(os.path.join(Path, 'data', self.run_id, 'models',
                                     '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel, Train=False, Path=None):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        if Train:
            num_batches = int((channel.y_train.shape[0] - self.config.l_s)
                              / self.config.batch_size)
        else:
            num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                              / self.config.batch_size)

        logger.debug("predict: num_batches ", num_batches)

        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))

        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                if Train:
                    idx = channel.y_test.shape[0]
                else:
                    idx = channel.y_train.shape[0]

            if Train:
                X_train_batch = channel.X_train[prior_idx:idx]
                y_hat_batch = self.model.predict(X_train_batch)
            else:
                X_test_batch = channel.X_test[prior_idx:idx]
                y_hat_batch = self.model.predict(X_test_batch)

            logger.debug("predict: batch ", i, " - ", y_hat_batch.shape)

            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        if Train:
            channel.y_train_hat = self.y_hat
        else:
            if self.config.FFT:
                logger.info('FFT modelling')
                channel.y_hat = sp.fft.irfft(self.y_hat)
            else:
                channel.y_hat = self.y_hat

        if Path is None:
            Path = ""

        np.save(os.path.join(Path, 'data', self.run_id, 'y_hat', '{}.npy'
                                   .format(self.chan_id)), self.y_hat)

        return channel
