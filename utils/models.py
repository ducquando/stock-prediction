# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import glob

# Import functions
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers import LSTM, Dropout, RepeatVector
from stocks.utils.helpers import min_max_normalize, denormalization, candlestick3D


class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0
    
    def on_batch_begin(self, batch, logs={}):
        max_len = 10
        if self.counter % max_len == 0:
            self.model.reset_states()
        self.counter += 1


class Stocks:
    def get_Xy_training(self):
        """
        Get data features and data labels for training
        """
        # Get stock
        stock = self.stock.copy()
        
        # Stock features (Open, Close, Low, High)
        index_feature = np.array([[i+1, i+5, i, i+3] for i in range(1, stock.shape[1], 6)]).flatten()

        # Stock labels (Open, Close) of the next window
        index_label = np.array([[i+1, i+5] for i in range(1, stock.shape[1], 6)]).flatten()

        # Split training features and labels
        periods = len(stock) - self.train_period - self.predict_period + 1
        X, y = np.zeros((periods, self.train_period, self.dim_feature)), np.zeros((periods, self.predict_period, self.dim_label))

        # Append values
        for i in range(periods):
            data_label, data_feature = [], []

            # Get a window_size time frame for data feature
            for j in range(self.train_period):
                data_feature.append([stock.iloc[i + j, k] for k in index_feature])

            for j in range(self.predict_period):
                data_label.append([stock.iloc[i + j + self.train_period, k] for k in index_label])

            # Append new data sample
            X[i] = np.array(data_feature).reshape(self.train_period, self.dim_feature)
            y[i] = np.array(data_label).reshape(self.predict_period, self.dim_label)
            
        return X, y
    
    
    def get_X_forecast(self):
        """
        Get data features and data labels for making predictions
        """
        # Get stock
        stock = self.stock.copy()
        
        # Stock features (Open, Close, Low, High)
        index_feature = np.array([[i+1, i+5, i, i+3] for i in range(1, stock.shape[1], 6)]).flatten()
        
        # Save future prediction features
        periods = len(stock) - self.train_period - self.predict_period + 1
        periods_future = len(stock) - self.train_period + 1
        X_future = np.zeros((self.predict_period, self.train_period, self.dim_feature))
        
        # Append values
        for i in range(periods, periods_future):
            data_feature = []
            
            # Get a window_size time frame for data feature
            for j in range(self.train_period):
                data_feature.append([stock.iloc[i + j, k] for k in index_feature])

            # Append new data sample
            X_future[i - periods] = np.array(data_feature).reshape(self.train_period, self.dim_feature)
        
        return X_future
    
    
    def train_val_test_split(self, X, y):
        """
        Split the given set into training, validation, and testing with 6/2/2 ratio
        """
        # Split data into train, val and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, shuffle = False)

        # Convert from lists to Numpy arrays for reshaping purpose
        X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
        y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

        # Reshape the numpy array to fit the neural network input shape requirement
        X_train = X_train.reshape(X_train.shape[0], self.train_period, self.dim_feature)
        X_val = X_val.reshape(X_val.shape[0], self.train_period, self.dim_feature)
        X_test = X_test.reshape(X_test.shape[0], self.train_period, self.dim_feature)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def train_model(self):
        """
        Train the model
        """
        # Get training data
        X_train, X_val, _, y_train, y_val, _ = self.train_val_test_split(self.X, self.y)
        
        # Normalize data
        X_train_norm, y_train_norm = min_max_normalize(X_train, y_train)
        X_val_norm, y_val_norm = min_max_normalize(X_val, y_val)
        
        # Model
        autoencoder = Sequential([
            # Encoder
            ## Many to many
            LSTM(50, return_sequences = True, stateful = True, batch_input_shape = (2, self.train_period, self.dim_feature)),
            Dropout(0.5),
            ## Many to many
            LSTM(50, return_sequences = True, stateful = True),
            Dropout(0.5),
            ## Many to one
            LSTM(50, stateful = True),
            
            # Decoder
            ## One to many
            RepeatVector(self.predict_period),
            ## Many to many
            LSTM(50, return_sequences = True, stateful = True),
            Dropout(0.5),
            ## Classifiers
            LSTM(self.dim_label, return_sequences = True, stateful = True)
        ], name = "LSTM_many_to_many")
        
        # Compile and train the model with Mean Squared Error loss function
        autoencoder.compile(optimizer = Adam(learning_rate = 1e-5), loss = 'mse', metrics = ['mse'])
        lstm_performance = autoencoder.fit(X_train_norm, y_train_norm, validation_data = (X_val_norm, y_val_norm), shuffle = False, epochs = 15, batch_size = 2, callbacks = [ResetStatesCallback()])
        self.model = autoencoder
        
        return lstm_performance
    
    
    ## Call functions ##
    
        
    def get_companies(self):
        """
        Get the list of available stock tickers
        """
        return self.companies
    

    def plot_model_loss(self, performance, path = ""):
        """
        Save model loss
        """
        # Draw figure
        fig = plt.figure(figsize = (10, 5))
        plt.plot(performance.history['loss']);
        plt.plot(performance.history['val_loss'])
        plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'], loc = 'upper right')
        
        # Save figure
        plt.savefig(f"{path}outputs/model_loss.jpg")
      
    
    def test(self, company, path = ""):
        """
        Perform testing
        """
        # Get company id
        company_id = self.companies.index(company)
        
        # Get training data
        _, _, X_test, _, _, y_test = self.train_val_test_split(self.X, self.y)
        
        # MinMax normalize the test data
        X_test_norm, y_test_norm = min_max_normalize(X_test, y_test)
        
        # Get prediction on the test data
        y_pred_norm = self.model.predict(X_test_norm, batch_size = 2)
        
        # Calculate the average MSE
        average_mse = mean_squared_error(y_pred_norm.flatten(), y_test_norm.flatten())
        
        # Convert the result back to stock price (i.e., de-normalization) for visualization purpose
        y_pred = denormalization(X_test, y_pred_norm)
        
        # Plot the subset splits
        fig, ax = plt.subplots(figsize = (10, 5))
        candlestick3D(ax, y_pred, company = company_id, colordown = 'blue', full = False)
        candlestick3D(ax, y_test, company = company_id, colordown = 'red', full = False)
        ax.set_title(f"{company}: Stock trend")
        plt.xlabel('Time (days)'); plt.ylabel('Price in $')
        
        # Save figure
        plt.savefig(f"{path}outputs/test_{company}.jpg")
        
        return average_mse
        
        
    def forecast(self, company, path = ""):
        """
        Predict on the future
        """
        # Get company id
        company_id = self.companies.index(company)
        
        # Get prediction data
        X_forecast = self.get_X_forecast()
        
        # MinMax normalize the data
        X_forecast_norm, _ = min_max_normalize(X_forecast, np.zeros((2,self.train_period,self.dim_label)))

        # Get prediction on 7 days into the future
        y_forecast_norm = autoencoder.predict(X_forecast_norm, batch_size = 2)

        # Convert the result back to stock price (i.e., de-normalization) for visualization purpose
        y_forecast_pred = denormalization(X_forecast_norm, y_forecast_norm)

        # Plot the subset splits
        fig, ax = plt.subplots(figsize = (10, 5))
        candlestick3D(ax, y_forecast_pred, company = company_id, colordown = 'blue', full = False)
        ax.set_title(f"{company}: Stock trend")
        plt.xlabel('Time (days)'); plt.ylabel('Price in $')
        
        # Save figure
        plt.savefig(f"{path}outputs/forecast_{company}.jpg")


class VietnamStocks(Stocks):
    def __init__(self, market, sectors, date = "2007-01-11", train = 30, test = 7, path = "", pre_trained = None):
        self.path = path + 'dataset/vn/'
        self.date, self.sectors = date, sectors
        self.train_period, self.predict_period = train, test # days
        self.markets = market if market == "UPCOM" else "UNX"
        self.dataset = "UpcomIndex" if market == "UPCOM" else "UNXIndex"
        self.stock, self.companies = self.get_stock()
        self.dim_feature, self.dim_label = int(self.stock.shape[1] / 6) * 4, int(self.stock.shape[1] / 6) * 2
        self.X, self.y = self.get_Xy_training()
        self.model = pre_trained
    
    
    def get_stock(self):
        """
        Get a dataframe contains the stocks of all companies in the given time period that matches the criteria
        """
        # Get companies given criteria
        tickers = pd.read_csv(f'{self.path}ticker-overview.csv')
        ticker = tickers.loc[(tickers['exchange'] == self.markets) & np.array([tickers['industryEn'] == i for i in self.sectors]).any()]["ticker"]
        
        # Check if we have that dataset
        stks_loc = f'{self.path}stock-historical-data/'
        stks, valid, data_frames = glob.glob(stks_loc + '*.csv'), [], []
        for stk in ticker:
            name = f'{stks_loc}{str(stk)}-{self.dataset}-History.csv'
            if name in stks:
                valid.append(stk)
                
        # Compile the list of stock datas that meet requirements
        for i in range(len(valid)):
            data = pd.read_csv(f'{stks_loc}{str(valid[i])}-{self.dataset}-History.csv')
            indexing = data.loc[data['TradingDate'] == self.date]
            if len(indexing) == 1:
                data = data.iloc[indexing.index[0]:,:]
                data = data[['TradingDate', 'Low', 'Open', 'Volume', 'High', 'Close', 'Close']]
                data_frames.append(data)
                
        # Merge into one
        stock = reduce(lambda left, right: pd.merge(left, right, on = ['TradingDate'], how = 'outer'), data_frames)
        cols = [[f"Low_{i}", f"Open_{i}", f"Volume_{i}", f"High_{i}", f"Close_{i}", f"Adjusted Close_{i}"] for i in range(1, int(stock.shape[1] / 6) + 1)]
        stock.columns = np.append(np.array("Date"), np.array(cols).flatten())
        
        # Drop nan values
        stock = stock.dropna()
        stock = stock.reset_index(drop = True)
        
        return stock, valid
    
        
class NasdaqStocks(Stocks):
    def __init__(self, sectors, date = "04-01-2007", train = 30, test = 7, path = "", pre_trained = None):
        self.path = path + 'dataset/nasdaq/'
        self.date, self.sectors = date, sectors
        self.train_period, self.predict_period = train, test # days
        self.stock, self.companies = self.get_stock()
        self.dim_feature, self.dim_label = int(self.stock.shape[1] / 6) * 4, int(self.stock.shape[1] / 6) * 2
        self.X, self.y = self.get_Xy_training()
        self.model = pre_trained
    
    
    def get_stock(self):
        """
        Get a dataframe contains the stocks of all companies in the given time period that matches the criteria
        """
        # Get companies given criteria
        tickers = pd.read_csv(f'{self.path}nasdaq-100.csv')
        ticker = tickers.loc[pd.Series(np.array([tickers['Sector'] == i for i in self.sectors]).flatten())]["Ticker"] 
        
        # Check if we have that dataset
        stks_loc = f'{self.path}stock-historical-data/'
        stks, valid, data_frames = glob.glob(stks_loc + '*.csv'), [], []
        for stk in ticker:
            name = f'{stks_loc}{str(stk)}.csv'
            if name in stks:
                valid.append(stk)
                
        # Compile the list of stock datas that meet requirements
        for i in range(len(valid)):
            data = pd.read_csv(f'{stks_loc}{str(valid[i])}.csv')
            indexing = data.loc[data['Date'] == self.date]
            if len(indexing) == 1:
                data = data.iloc[indexing.index[0]:,:]
                data_frames.append(data)
                
        # Merge into one
        stock = reduce(lambda left, right: pd.merge(left, right, on = ['TradingDate'], how = 'outer'), data_frames)
        cols = [[f"Low_{i}", f"Open_{i}", f"Volume_{i}", f"High_{i}", f"Close_{i}", f"Adjusted Close_{i}"] for i in range(1, int(stock.shape[1] / 6) + 1)]
        stock.columns = np.append(np.array("Date"), np.array(cols).flatten())
        
        # Drop nan values
        stock = stock.dropna()
        stock = stock.reset_index(drop = True)
        
        return stock, valid
