# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
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
    def __init__(self, market, sectors, date, train, test, path, pre_trained):
        self.path, self.market = path, market
        self.date, self.sectors = date, sectors
        self.train_period, self.predict_period = train, test
        self.stock, self.companies = self.get_stock()
        self.dim_feature, self.dim_label = int(self.stock.shape[1] / 6) * 4, int(self.stock.shape[1] / 6) * 2
        self.X, self.y = self.get_Xy_training()
        self.model, self.performance, self.pre_trained = None, None, pre_trained
        self.forcast = None

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
    
    
    def train_val_test_split(self, X_input, y_input):
        """
        Split the given set into training, validation, and testing with 6/2/2 ratio
        """
        X = X_input.copy()
        y = y_input.copy()
        
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
    
        
    ## Call functions ##
    
    
    def init_model(self):
        """
        Initiate model by loading pre-trained model. If cannot, training new model.
        """
        if self.pre_trained:
            model = self.load_model()
        else:
            model = self.train_model()
            self.save_model()
        
        return model
        
        
    def load_model(self):
        """
        Load pre-trained model
        """
        # Get model
        name = ' '.join(map(str, self.sectors))
        path = "{self.path}outputs/models/{self.market}/{name}.h5"
        autoencoder = keras.models.load_model(path)
        
        # Load model into object
        self.model = autoencoder
        
        return autoencoder
        
    
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
            LSTM(100, return_sequences = True, stateful = True),
            Dropout(0.5),
            ## Many to one
            LSTM(100, stateful = True),
            
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
        performance = autoencoder.fit(X_train_norm, y_train_norm, validation_data = (X_val_norm, y_val_norm), shuffle = False, epochs = 15, batch_size = 2, callbacks = [ResetStatesCallback()])
        
        # Load model into object
        self.model = autoencoder
        self.performance = performance
        
        return autoencoder
    
    
    def save_model(self):
        """
        Save trained model
        """
        # Get model
        autoencoder = self.model
        
        # Save model
        name = ' '.join(map(str, self.sectors))
        path = f"{self.path}outputs/models/{self.market}/{name}.h5"
        autoencoder.save(path)
        
        
    def get_companies(self):
        """
        Get the list of available stock tickers
        """
        return self.companies
    

    def plot_model_loss(self):
        """
        Save model loss
        """
        name = ' '.join(map(str, self.sectors))
        if self.pre_trained:
            img = cv.imread(f"{self.path}outputs/models/{self.market}/{name}model_loss.jpg", cv2.IMREAD_COLOR)
            img = cv.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        else:
            # Get model performance
            performance = self.performance
            
            # Draw figure
            fig = plt.figure(figsize = (10, 5))
            plt.plot(performance.history['loss']);
            plt.plot(performance.history['val_loss'])
            plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.legend(['Training', 'Validation'], loc = 'upper right')
            
            # Save figure
            plt.savefig(f"{self.path}outputs/models/{self.market}/{name}model_loss.jpg")
      
    
    def test(self, company, currency = "$"):
        """
        Perform testing
        """
        # Get company id
        company_id = self.companies.index(company)
        
        # Get training data and model
        autoencoder = self.model
        _, _, X_test, _, _, y_test = self.train_val_test_split(self.X, self.y)
        
        # MinMax normalize the test data
        X_test_norm, y_test_norm = min_max_normalize(X_test, y_test)
        
        # Get prediction on the test data
        y_pred_norm = autoencoder.predict(X_test_norm, batch_size = 2)
        
        # Calculate the average MSE
        average_mse = mean_squared_error(y_pred_norm.flatten(), y_test_norm.flatten())
        
        # Convert the result back to stock price (i.e., de-normalization) for visualization purpose
        y_pred = denormalization(X_test, y_pred_norm)
        
        # Plot the subset splits
        fig, ax = plt.subplots(figsize = (10, 5))
        candlestick3D(ax, y_pred, company = company_id, colordown = 'blue', full = False)
        candlestick3D(ax, y_test, company = company_id, colordown = 'red', full = False)
        ax.set_title(f"{company}: Stock trend")
        plt.xlabel('Time (days)'); plt.ylabel(f'Price ({currency})')
        
        # Save figure
        plt.savefig(f"{self.path}outputs/test_{company}.jpg")
        
        return y_pred, average_mse
        
        
    def forecast(self, company, currency = "$"):
        """
        Predict on the future
        """
        # Get company id
        company_id = self.companies.index(company)
        
        # Get prediction data and model
        autoencoder = self.model
        X_forecast = self.get_X_forecast()
        
        # MinMax normalize the data
        X_forecast_norm, _ = min_max_normalize(X_forecast, np.zeros((2,self.train_period,self.dim_label)))

        # Get prediction on 7 days into the future
        y_forecast_norm = autoencoder.predict(X_forecast_norm, batch_size = 2)

        # Convert the result back to stock price and save it
        y_forecast = denormalization(X_forecast, y_forecast_norm)
        self.forcast = y_forecast

        # Plot the subset splits
        fig, ax = plt.subplots(figsize = (10, 5))
        candlestick3D(ax, y_forecast, company = company_id, colordown = 'blue', full = False)
        ax.set_title(f"{company}: Stock trend")
        plt.xlabel('Time (days)'); plt.ylabel(f'Price ({currency})')
        
        # Save figure
        plt.savefig(f"{self.path}outputs/forecast_{company}.jpg")
        
        return y_forecast
        
        
    def get_portfolio(self):
        """
        Get the list of companies to hold either when investors are risk-taking or prudent
        """
        # Get stock value
        y, companies = self.companies, self.forcast
        risky, prudent, throwing, keeping = {}, {}, {}, {}
        
        # Categorize into risky & prudent
        for company_id in range(len(companies)):
            # Convert to np.ndarray
            stk = np.array(y[:,0,company_id:company_id+2])
            stk_mean = np.array([(x[0]+x[1])/2 for x in stk]).flatten()
            
            # Get low and high time
            low, high = stk.copy().flatten().argmin(), stk.copy().flatten().argmax()
            low_day, high_day = int(low // 2), int(high // 2)
            
            # Get trend
            trend_mon = round(stk_mean[-1] - stk_mean[0], 3)
            trend_per = (trend_mon/stk_mean[0] + 1) * abs(high_day - low_day) / 7
            direction = True if trend_mon > 0 else False
            
            # Get recommendations
            price_buy, price_sell = stk[low_day][low_when[0]], stk[high_day][high_when[0]]
            price_profit = price_sell - price_buy
            if not direction:
                price_risk = price_sell * trend_per - price_buy if high_day < low_day else price_buy - price_buy * trend_per
                throwing[companies[company_id]] = round(price_profit, 3)
            else:
                price_risk = price_sell - price_buy * trend_per if high_day < low_day else price_sell - price_sell * trend_per
                keeping[companies[company_id]] = round(price_profit, 3)

            if price_risk > price_profit:
                risky[companies[company_id]] = round(price_profit, 3)
            else:
                prudent[companies[company_id]] = round(price_profit, 3)
                
        # Sort by profit
        sorted_risky = {list(risky.keys())[i]: list(risky.values())[i] for i in np.argsort(list(risky.values()))[-1]}   # DESC
        sorted_prudent = {list(prudent.keys())[i]: list(prudent.values())[i] for i in np.argsort(list(prudent.values()))[-1]}   # DESC
        sorted_keeping = {list(keeping.keys())[i]: list(keeping.values())[i] for i in np.argsort(list(keeping.values()))}   # ASC
        sorted_throwing = {list(throwing.keys())[i]: list(throwing.values())[i] for i in np.argsort(list(throwing.values()))}   # ASC

        # Print portfolio management
        print(f"Portfolio management tips for the next 7 days:")
        print(f"  - Should hold: {sorted_keeping}")
        print(f"  - Should sell: {sorted_throwing}")
        print(f"  - Risk takers: {sorted_risky}")
        print(f"  - Safety net: {sorted_prudent}")
        
    
    def get_statistics(self, company, currency = "$"):
        """
        Get stock statistics
        """
        # Get company id and stock values
        company_id = self.companies.index(company)
        y = self.forcast

        # Convert to np.ndarray
        stk = np.array(y[:,0,company_id:company_id+2])
        stk_mean = np.array([(x[0]+x[1])/2 for x in stk]).flatten()
        
        # Get descriptive values
        current = round(stk[0][0], 3)
        low = round(stk.copy().flatten().min(), 3)
        high = round(stk.copy().flatten().max(), 3)
        avg = round(stk_mean.mean(), 3)
        trend_mon = round(stk_mean[-1] - stk_mean[0], 3)
        trend_per = round(trend_mon/stk_mean[0]*100, 3)
        direction = "Increase" if trend_mon > 0 else "Decrease"
        
        # Print stock stats
        print(f"Stock statistics for {company} in the next 7 days:")
        print(f"  - Current price: {currency}{current}/share")
        print(f"  - Lowest price: {currency}{low}/share")
        print(f"  - Average price: {currency}{avg}/share")
        print(f"  - Highest price: {currency}{high}/share")
        print(f"  - Trend: {direction} {currency}{trend_mon}/share ({trend_per}%)")
        
        
    def get_recommendation(self, company, currency = "$"):
        """
        Get recommendation for trading stocks
        """
        # Get company id and stock values
        company_id = self.companies.index(company)
        y = self.forcast

        # Convert to np.ndarray
        stk = np.array(y[:,0,company_id:company_id+2])
        stk_mean = np.array([(x[0]+x[1])/2 for x in stk]).flatten()
        
        # Get low time
        low = stk.copy().flatten().argmin()
        low_day = int(low // 2)
        low_when = [0, "opening"] if stk[low_day][0] < stk[low_day][1] else [1, "close"]
        
        # Get high time
        high = stk.copy().flatten().argmax()
        high_day = int(high // 2)
        high_when = [0, "opening"] if stk[high_day][0] > stk[high_day][1] else [1, "close"]
        
        # Get trend
        trend_mon = round(stk_mean[-1] - stk_mean[0], 3)
        trend_per = (trend_mon/stk_mean[0] + 1) * abs(high_day - low_day) / 7
        direction = "Increase" if trend_mon > 0 else "Decrease"
        
        # Get recommendations
        price_buy = round(stk[low_day][low_when[0]], 3)
        price_sell = round(stk[high_day][high_when[0]], 3)
        price_profit = round(price_sell - price_buy, 3)
        if direction == "Decrease":
            sell = "right now" if high_day == 0 else f"on {high_when[1]} of the {high_day}-th day"
            buy = "later" if low_day == stk.shape[0] else f"on {low_when[1]} of the {low_day}-th day"
            price_risk = price_sell * trend_per - price_buy if high_day < low_day else price_buy - price_buy * trend_per
        elif direction == "Increase":
            buy = "right now" if low_day == 0 else f"on {low_when[1]} of the {low_day}-th day"
            sell = "later" if high_day == stk.shape[0] else f"on {high_when[1]} of the {high_day}-th day"
            price_risk = price_sell - price_buy * trend_per if high_day < low_day else price_sell - price_sell * trend_per
        price_risk = round(price_risk, 3)
        conclusion = "RISKY!" if price_risk > price_profit else "The risk is acceptable, you can buy some shares"

        # Print stock recommendations
        print(f"Trading recommendation for {company} in the next 7 days:")
        print(f"  - Best buying: {currency}{price_buy}/share {buy}")
        print(f"  - Best selling: {currency}{price_sell}/share {sell}")
        print(f"  - Trading profit: {currency}{price_profit}/share")
        print(f"  - Trading risk: {currency}{price_risk}/share")
        print(f"=> Conclusion: {conclusion}")
        
        return price_profit, price_risk
        

class VietnamStocks(Stocks):
    def __init__(self, market, sectors, date = "2007-01-11", train = 30, test = 7, path = "", pre_trained = False):
        this_market = market if market == "UPCOM" else "UNX"
        self.dataset = "UpcomIndex" if market == "UPCOM" else "UNXIndex"
        Stocks.__init__(self, this_market, sectors, date, train, test, path, pre_trained)
    
    
    def get_stock(self):
        """
        Get a dataframe contains the stocks of all companies in the given time period that matches the criteria
        """
        # Get companies given criteria
        path = self.path + 'dataset/vn/'
        tickers = pd.read_csv(f'{path}ticker-overview.csv')
        ticker = tickers.loc[(tickers['exchange'] == self.market) & np.array([tickers['industryEn'] == i for i in self.sectors]).any()]["ticker"]
        
        # Check if we have that dataset
        stks_loc = f'{path}stock-historical-data/'
        stks, valid, dfs, companies = glob.glob(stks_loc + '*.csv'), [], [], []
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
                dfs.append(data)
                companies.append(valid[i])
                
        # Merge into one
        stock = reduce(lambda left, right: pd.merge(left, right, on = ['TradingDate'], how = 'outer'), dfs)
        cols = [[f"Low_{i}", f"Open_{i}", f"Volume_{i}", f"High_{i}", f"Close_{i}", f"Adjusted Close_{i}"] for i in range(1, int(stock.shape[1] / 6) + 1)]
        stock.columns = np.append(np.array("Date"), np.array(cols).flatten())
        
        # Drop nan values
        stock = stock.dropna()
        stock = stock.reset_index(drop = True)
        
        return stock, companies
    
        
class NasdaqStocks(Stocks):
    def __init__(self, sectors, date = "04-01-2007", train = 30, test = 7, path = "", pre_trained = False):
        Stocks.__init__(self, "NASDAQ", sectors, date, train, test, path, pre_trained)
    
    def get_stock(self):
        """
        Get a dataframe contains the stocks of all companies in the given time period that matches the criteria
        """
        # Get companies given criteria
        path = self.path + 'dataset/nasdaq/'
        tickers = pd.read_csv(f'{path}nasdaq-100.csv')
        ticker = tickers.loc[pd.Series(np.array([tickers['Sector'] == i for i in self.sectors]).flatten())]["Ticker"] 
        
        # Check if we have that dataset
        stks_loc = f'{path}stock-historical-data/'
        stks, valid, dfs, companies = glob.glob(stks_loc + '*.csv'), [], [], []
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
                dfs.append(data)
                companies.append(valid[i])
                
        # Merge into one
        stock = reduce(lambda left, right: pd.merge(left, right, on = ['Date'], how = 'outer'), dfs)
        cols = [[f"Low_{i}", f"Open_{i}", f"Volume_{i}", f"High_{i}", f"Close_{i}", f"Adjusted Close_{i}"] for i in range(1, int(stock.shape[1] / 6) + 1)]
        stock.columns = np.append(np.array("Date"), np.array(cols).flatten())
        
        # Drop nan values
        stock = stock.dropna()
        stock = stock.reset_index(drop = True)
        
        return stock, companies
