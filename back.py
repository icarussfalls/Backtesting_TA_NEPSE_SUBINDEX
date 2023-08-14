import pandas as pd
import numpy as np

def aggregate_ohlc_data(daily_data):
    """
    Aggregate daily OHLCV data into weekly, monthly, and yearly data.

    Parameters:
        daily_data (DataFrame): DataFrame containing daily OHLCV data with 'Date', 'Open', 'High', 'Low', 'Close', and 'Volume' columns.

    Returns:
        Tuple of DataFrames: (weekly_data, monthly_data, yearly_data)
    """
    # Make sure 'Date' column is in datetime format
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])

    # Set the 'Date' column as the index and create copies of the data for each resampling
    daily_data_copy = daily_data.copy()
    weekly_data = daily_data_copy.set_index('Date').resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    daily_data_copy = daily_data.copy()
    monthly_data = daily_data_copy.set_index('Date').resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    daily_data_copy = daily_data.copy()
    yearly_data = daily_data_copy.set_index('Date').resample('Y').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    return weekly_data, monthly_data, yearly_data


daily_data = pd.read_csv('HOTELS.csv')
#daily_data = clean_ohlc_data(daily_data)
# Assuming you have a DataFrame called 'daily_data' with columns 'Date', 'Open', 'High', 'Low', 'Close', and 'Volume'
weekly_data, monthly_data, yearly_data = aggregate_ohlc_data(daily_data)

data = weekly_data
data = data.sort_values(by='Date', ascending=True)
#data = data.set_index('Date')

# Functions to calculate technical indicators
def calculate_sma(data, window):
    return data['Close'].rolling(window=window, min_periods=1).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)
    rolling_std = data['Close'].rolling(window=window, min_periods=1).std()
    upper_band = sma + num_std * rolling_std
    lower_band = sma - num_std * rolling_std
    return upper_band, lower_band

def calculate_stochastic_oscillator(data, window=14):
    highest_high = data['High'].rolling(window=window, min_periods=1).max()
    lowest_low = data['Low'].rolling(window=window, min_periods=1).min()
    stoch_k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()
    return stoch_k, stoch_d


# Function to generate signals for Simple Moving Averages (SMA) strategy
def generate_sma_signals(data, short_window=30, long_window=50):
    signals = pd.DataFrame(index=data.index)
    signals['SMA_Short'] = calculate_sma(data, window=short_window)
    signals['SMA_Long'] = calculate_sma(data, window=long_window)

    # Generate buy signals when SMA_Short crosses above SMA_Long
    signals['Buy_Signal'] = np.where((signals['SMA_Short'] > signals['SMA_Long']) & (signals['SMA_Short'].shift() <= signals['SMA_Long'].shift()), 1, 0)

    # Generate sell signals when SMA_Short crosses below SMA_Long
    signals['Sell_Signal'] = np.where((signals['SMA_Short'] < signals['SMA_Long']) & (signals['SMA_Short'].shift() >= signals['SMA_Long'].shift()), 1, 0)

    return signals


# Function to generate signals for Moving Average Convergence Divergence (MACD) strategy
def generate_macd_signals(data, short_window=12, long_window=26, signal_window=9):
    signals = pd.DataFrame(index=data.index)
    signals['MACD'], signals['Signal_Line'], signals['MACD_Histogram'] = calculate_macd(data)

    # Generate buy signals when MACD crosses above Signal_Line
    signals['Buy_Signal'] = np.where((signals['MACD'] > signals['Signal_Line']) & (signals['MACD'].shift() <= signals['Signal_Line'].shift()), 1, 0)

    # Generate sell signals when MACD crosses below Signal_Line
    signals['Sell_Signal'] = np.where((signals['MACD'] < signals['Signal_Line']) & (signals['MACD'].shift() >= signals['Signal_Line'].shift()), 1, 0)

    return signals


# Function to generate signals for Relative Strength Index (RSI) strategy
def generate_rsi_signals(data, rsi_oversold=30, rsi_overbought=70):
    signals = pd.DataFrame(index=data.index)
    signals['RSI'] = calculate_rsi(data)

    # Generate buy signals when RSI crosses above the oversold threshold
    signals['Buy_Signal'] = np.where((signals['RSI'] > rsi_oversold) & (signals['RSI'].shift() <= rsi_oversold), 1, 0)

    # Generate sell signals when RSI crosses below the overbought threshold
    signals['Sell_Signal'] = np.where((signals['RSI'] < rsi_overbought) & (signals['RSI'].shift() >= rsi_overbought), 1, 0)

    return signals


# Function to generate signals for Bollinger Bands strategy
def generate_bollinger_bands_signals(data, window=20, num_std=2):
    signals = pd.DataFrame(index=data.index)
    upper_band, lower_band = calculate_bollinger_bands(data, window=window, num_std=num_std)

    # Generate buy signals when price crosses below the lower Bollinger Band
    signals['Buy_Signal'] = np.where((data['Close'] < lower_band) & (data['Close'].shift() >= lower_band.shift()), 1, 0)

    # Generate sell signals when price crosses above the upper Bollinger Band
    signals['Sell_Signal'] = np.where((data['Close'] > upper_band) & (data['Close'].shift() <= upper_band.shift()), 1, 0)

    return signals


# Function to generate signals for Stochastic Oscillator strategy
def generate_stochastic_signals(data, stoch_k_threshold=20, stoch_d_threshold=20):
    signals = pd.DataFrame(index=data.index)
    signals['%K'], signals['%D'] = calculate_stochastic_oscillator(data)

    # Generate buy signals when %K and %D cross above the thresholds
    signals['Buy_Signal'] = np.where(((signals['%K'] > stoch_k_threshold) & (signals['%K'].shift() <= stoch_k_threshold)) &
                                     ((signals['%D'] > stoch_d_threshold) & (signals['%D'].shift() <= stoch_d_threshold)), 1, 0)

    # Generate sell signals when %K and %D cross below the thresholds
    signals['Sell_Signal'] = np.where(((signals['%K'] < stoch_k_threshold) & (signals['%K'].shift() >= stoch_k_threshold)) &
                                      ((signals['%D'] < stoch_d_threshold) & (signals['%D'].shift() >= stoch_d_threshold)), 1, 0)

    return signals



# Generate signals for all strategies
signals_sma = generate_sma_signals(data)
#print(signals_sma.loc[signals_sma['Buy_Signal'] == 1])
signals_rsi = generate_rsi_signals(data)
signals_macd = generate_macd_signals(data)
signals_bb = generate_bollinger_bands_signals(data)
signals_stoch = generate_stochastic_signals(data)

signals_sma['Position'] = 0
signals_rsi['Position'] = 0
signals_macd['Position'] = 0
signals_bb['Position'] = 0
signals_stoch['Position'] = 0


def backtest_strategy(strategy_name, data, signals, starting_portfolio_value=100000, min_holding_period=1, trading_fee=0.004):
    print(f"Backtesting {strategy_name} Strategy...")
    signals = signals.dropna()

    # Create a copy of the DataFrame to avoid modifying the original
    signals_copy = signals.copy()

    # Initialize a holding period counter and entry price variables
    holding_period = 0
    entry_price = 0

    # Initialize entry and exit dates as empty lists
    entry_dates = []
    exit_dates = []
    returns_list = []  # Store individual trade returns

    # Backtesting loop
    in_position = False  # Flag to indicate if a position is open

    for i in range(len(signals_copy)):
        if signals_copy['Buy_Signal'][i] == 1:
            if not in_position:  # If not already in a position, enter a long position
                signals_copy.loc[signals_copy.index[i], 'Position'] = 1  # Enter long position
                holding_period = 1  # Set the holding period to 1 as we just entered the position
                entry_dates.append(signals_copy.index[i])  # Append Entry Date
                entry_price = data.loc[signals_copy.index[i], 'Close']  # Store the entry price
                in_position = True  # Set the flag to indicate we are in a position

        if in_position:  # If in a position, check for exit signal
            if signals_copy['Sell_Signal'][i] == 1:
                # Check if minimum holding period is satisfied after buying
                if holding_period >= min_holding_period:
                    signals_copy.loc[signals_copy.index[i], 'Position'] = 0  # Exit long position
                    holding_period = 0  # Reset the holding period counter
                    exit_dates.append(signals_copy.index[i])  # Append Exit Date
                    exit_price = data.loc[signals_copy.index[i], 'Close']  # Store the exit price
                    in_position = False  # Set the flag to indicate we are no longer in a position

                    # Calculate and store the individual trade return with trading fees
                    returns = (exit_price / entry_price) * (1 - trading_fee) - 1
                    returns_list.append(returns)

        # Increment the holding period counter if already in a position
        if in_position:
            holding_period += 1

    # Append the last entry and exit dates if the last trade is still open at the end of the data
    if in_position:
        exit_dates.append(signals_copy.index[-1])
        exit_price = data.iloc[-1]['Close']
        returns = (exit_price / entry_price) * (1 - trading_fee) - 1
        returns_list.append(returns)

    # Calculate the average holding period
    holding_periods = [(exit_date - entry_date).days for entry_date, exit_date in zip(entry_dates, exit_dates)]
    average_holding_period = sum(holding_periods) / len(holding_periods)

    # Create a series of strategy returns using entry and exit dates
    strategy_returns = pd.Series(returns_list, index=entry_dates)

    # Combine daily returns with the signals DataFrame for analysis
    results = data[['Open', 'High', 'Low', 'Close']].join(strategy_returns.rename(f'{strategy_name} Returns'))

    # Calculate strategy performance metrics
    total_return = strategy_returns.sum()
    trading_days_per_year = 52

    # Calculate annualized return
    if total_return >= 0:
        annualized_return = (1 + total_return) ** (trading_days_per_year / len(data)) - 1
    else:
        positive_annualized_return = (1 + abs(total_return)) ** (trading_days_per_year / len(data)) - 1
        annualized_return = -positive_annualized_return




    # Calculate final portfolio value using cumulative product of strategy returns
    final_portfolio_value = starting_portfolio_value * (1 + strategy_returns).cumprod().iloc[-1]

    # Calculate mean and standard deviation
    mean_returns = np.mean(strategy_returns)
    std_returns = np.std(strategy_returns)

    # Normalize the data
    normalized_returns = (strategy_returns - mean_returns) / std_returns

    # Calculate the standard deviation of the normalized returns
    strategy_std = np.std(normalized_returns)

    # Calculate the total time period in years
    total_time_years = len(strategy_returns) / trading_days_per_year

    # Calculate annualized volatility
    annualized_volatility = strategy_std * np.sqrt(total_time_years)

    # Define risk-free rate
    risk_free_rate = 0.03

    # Calculate daily risk-free rate
    risk_free_rate_daily = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1

    # Calculate average return of the strategy
    average_return = np.mean(strategy_returns)

    # Calculate excess return over risk-free rate
    excess_return = average_return - risk_free_rate_daily

    # Calculate Sharpe Ratio
    sharpe_ratio = excess_return / annualized_volatility

    # Calculate Buy and Hold Return
    buy_and_hold_return = (data['Close'].pct_change().fillna(0) + 1).cumprod().iloc[-1] - 1

    # Calculate total trades and winning trades
    total_trades = len(entry_dates)
    winning_trades = sum(strategy_returns > 0)

    # Calculate win percentage
    win_percentage = winning_trades / total_trades

    # Calculate positive and negative returns
    positive_returns = strategy_returns[strategy_returns > 0]
    negative_returns = strategy_returns[strategy_returns < 0]

    # Calculate profit factor
    profit_factor = positive_returns.sum() / abs(negative_returns.sum())

    # Calculate cumulative returns, drawdown, and maximum drawdown
    cumulative_returns = (strategy_returns + 1).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = cumulative_returns / peak - 1
    max_drawdown = drawdown.min()



    
    # Create a pandas Series with the performance metrics
    performance_metrics = pd.Series({
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Buy and Hold Return': buy_and_hold_return,
        'Average Holding Period': average_holding_period,
        'Win Percentage': win_percentage,
        'Profit Factor': profit_factor,
        'Maximum Drawdown': max_drawdown,
        'Total Trades': total_trades
    }, name=strategy_name)

    # Return the performance metrics as a Series
    return performance_metrics

# Backtest and print performance metrics for each individual strategy
backtest_strategy("Simple Moving Averages (SMA)", data, signals_sma)
backtest_strategy("Relative Strength Index (RSI)", data, signals_rsi)
backtest_strategy("Moving Average Convergence Divergence (MACD)", data, signals_macd)
backtest_strategy("Bollinger Bands (BB)", data, signals_bb)
backtest_strategy("Stochastic Oscillator", data, signals_stoch)


# Perform backtesting and store results in a list of DataFrames
results = []
results.append(backtest_strategy("Simple Moving Averages (SMA)", data, signals_sma))
results.append(backtest_strategy("Relative Strength Index (RSI)", data, signals_rsi))
results.append(backtest_strategy("Moving Average Convergence Divergence (MACD)", data, signals_macd))
results.append(backtest_strategy("Bollinger Bands (BB)", data, signals_bb))
results.append(backtest_strategy("Stochastic Oscillator", data, signals_stoch))

# Concatenate the results into a final DataFrame
final_results = pd.concat(results, axis=1)

# Print the final DataFrame with strategies as columns and metrics as rows
print(final_results)
final_results.to_excel("backtest_results.xlsx", index=True)
