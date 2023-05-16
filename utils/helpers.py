# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

def min_max_normalize(Xs, ys):
    """
    MinMax normalize the validation data:
        x = (x - min(x) / (max(x) - min(x))

    Params:
        Xs (np.ndarray): 3D array (samples, win, features)
        ys (np.ndarray): 3D array (samples, win, features)
    """
    # Calculate min and max
    X_min = np.array([np.min(Xs[:,:,i]) for i in range(Xs.shape[2])]).reshape(1,1,Xs.shape[2])
    X_max = np.array([np.max(Xs[:,:,i]) for i in range(Xs.shape[2])]).reshape(1,1,Xs.shape[2])

    # Reshape
    z_dimension = np.array([[i, i+1] for i in range(0, Xs.shape[2], 4)]).flatten()
    y_min, y_max = X_min[:,:,z_dimension], X_max[:,:,z_dimension]
    
    # Normalize X and y
    X_norm = (Xs - X_min) / (X_max - X_min)
    y_norm = (ys - y_min) / (y_max - y_min)

    return X_norm, y_norm


def denormalization(X_norm, y_norm):
    """
    Convert the result back to stock price for visualization purpose:
        x = x * (max(x) - min(x)) + min(x)

    Params:
        Xs (np.ndarray): 3D array (samples, win, features)
        ys (np.ndarray): 3D array (samples, win, features)
    """
    # Calculate min and max and perform reshaping
    X_min = np.array([np.min(X_norm[:,:,i]) for i in range(X_norm.shape[2])]).reshape(1,1,X_norm.shape[2])
    X_max = np.array([np.max(X_norm[:,:,i]) for i in range(X_norm.shape[2])]).reshape(1,1,X_norm.shape[2])

    # Denormalize y with respect to x
    z_dimension = np.array([[i, i+1] for i in range(0, X_norm.shape[2], 4)]).flatten()
    ys = y_norm * (X_max[:,:,z_dimension] - X_min[:,:,z_dimension]) + X_min[:,:,z_dimension]

    return ys

def candlestick3D(ax, stocks, company = 0, begin_x = 0, width = 0.2, colorup = 'k',
                  colordown = 'r', ochl = True, linewidth = 0.5, full = True):
    """
    Plot the stock prices using Western candle stick bar. If close >= open,
    use colorup to color the bar, otherwise use colordown.

    Parameters:
      ax (Axes): an Axes instance to plot to
      stocks (list): array of stock data ()
      width (float): fraction of a day for the open and close lines
      colorup (color): the color of the lines close >= open
      colordown (color): the color of the lines where close <  open
      ochl (bool):  argument to select between ochl and ohlc ordering of quotes
      linewidth (float): linewidth of lines

    Returns:
      ret (lines, openlines, closelines)
    """
    OFFSET = width / 2.0
    lines, openlines, closelines = [], [], []
    company = 4 * company if full else 2 * company

    # Get stock for each open date
    for i in range(stocks.shape[0]):
        # Stock informations (Open, Close, Low, High)
        open, close = stocks[i][0][company + 0], stocks[i][0][company + 1]
        if full:
            low, high = stocks[i][0][company + 2], stocks[i][0][company + 3]
        else:
            low, high = (open, close) if open < close else (open, close)

        # Set and x-coordinate and color according to that day's stock trend
        color = colorup if close >= open else colordown
        x = begin_x + i

        # Draw low -> high lines
        vline = Line2D(xdata = (x, x), ydata = (low, high), color = color, linewidth = linewidth)
        lines.append(vline)

        # Draw open lines
        openline = Line2D(xdata = (x - OFFSET, x), ydata = (open, open),
                        color = color, linewidth = linewidth)
        openlines.append(openline)

        # Draw close lines
        closeline = Line2D(xdata = (x , x + OFFSET), ydata = (close, close),
                         color = color, linewidth = linewidth)
        closelines.append(closeline)

        ax.add_line(vline)
        ax.add_line(openline)
        ax.add_line(closeline)

    ax.autoscale_view()

    return lines, openlines, closelines


def candlestick2D(ax, stock, company = 0, begin_x = 0, width = 0.2, colorup = 'k',
                  colordown = 'r', ochl = True, linewidth = 0.5, full = True):
    """
    Plot the stock prices using Western candle stick bar. If close >= open,
    use colorup to color the bar, otherwise use colordown.

    Parameters:
      ax (Axes): an Axes instance to plot to
      stock (list): 2D array of stock data
      width (float): fraction of a day for the open and close lines
      colorup (color): the color of the lines close >= open
      colordown (color): the color of the lines where close <  open
      ochl (bool):  argument to select between ochl and ohlc ordering of quotes
      linewidth (float): linewidth of lines

    Returns:
      ret (lines, openlines, closelines)
    """
    OFFSET = width / 2.0
    lines, openlines, closelines = [], [], []
    company = 4 * company if full else 2 * company

    # Get stock for each open date
    for i in range(stock.shape[0]):
        # Stock informations (Open, Adjusted Close, Low, High)
        open, close = stock[i][company + 0], stock[i][company + 1]
        if full:
            low, high = stock[i][company + 2], stock[i][company + 3]
        else:
            low, high = (open, close) if open < close else (open, close)

        # Set and x-coordinate and color according to that day's stock trend
        color = colorup if close >= open else colordown
        x = begin_x + i

        # Draw low -> high lines
        vline = Line2D(xdata = (x, x), ydata = (low, high), color = color, linewidth = linewidth)
        lines.append(vline)

        # Draw open lines
        openline = Line2D(xdata = (x - OFFSET, x), ydata = (open, open),
                        color = color, linewidth = linewidth)
        openlines.append(openline)

        # Draw close lines
        closeline = Line2D(xdata = (x , x + OFFSET), ydata = (close, close),
                         color = color, linewidth = linewidth)
        closelines.append(closeline)

        ax.add_line(vline)
        ax.add_line(openline)
        ax.add_line(closeline)

    ax.autoscale_view()

    return lines, openlines, closelines