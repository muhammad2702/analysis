import ccxt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import asyncio
import websockets
import json
from collections import deque
from datetime import datetime
import os 
plots_directory = 'plots'
import requests

# Your Pushbullet access token
PUSHBULLET_TOKEN = 'o.MUbXXeMC13fYmFOXP9B8fJd7CR4Icdqh'

def send_push_notification(title, message):
    """Send a push notification to your phone via Pushbullet."""
    url = 'https://api.pushbullet.com/v2/pushes'
    headers = {
        'Access-Token': PUSHBULLET_TOKEN,
        'Content-Type': 'application/json',
    }
    payload = {
        'type': 'note',
        'title': title,
        'body': message,
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Push notification sent successfully.")
    else:
        print("Failed to send notification:", response.status_code, response.text)


send_push_notification("APP", "APP RUN")
        
def orderbook():
    binance = ccxt.binance()
    while(True):
        orderbook = binance.fetch_order_book('BTC/USDT')

        sum_size = np.sum( orderbook['bids'][i][1] for i in range (0, 5))
        maxi = 0
        mini = 0
        prices = pd.DataFrame()
        sizes  =  pd.DataFrame()
        prices['bids'] = 0
        sizes['bids_size'] = 0


        for j in range(0, (len(orderbook['bids']))):
                    prices.at[j , 'bids'] = orderbook['bids'][j][0]
                    sizes.at[j , 'bids_size'] = orderbook['bids'][j][1]
                    if (orderbook['bids'][j-1][0] <  orderbook['bids'][j][0]):
                        maxi = orderbook['bids'][j][0]
                    if (orderbook['bids'][j-1][0] >  orderbook['bids'][j][0]):
                        mini = orderbook['bids'][j][0] 
        rangebid  = maxi - mini       
        print(f'Bid Size : {sum_size}'  )
        print(f"Bid range : {rangebid}")




        prices['asks'] = 0
        sizes['asks_size'] = 0





        sum_size_asks = np.sum( orderbook['asks'][i][1] for i in range (0, 5))
        maxi_asks = 0
        mini_asks = 0

        for j in range(0, (len(orderbook['asks']))):
                    prices.at[j , 'asks'] = orderbook['asks'][j][0]
                    sizes.at[j , 'asks_size'] = orderbook['asks'][j][1]

                    if (orderbook['asks'][j-1][0] <  orderbook['asks'][j][0]):
                        maxi_asks = orderbook['asks'][j][0]
                    if (orderbook['asks'][j-1][0] >  orderbook['asks'][j][0]):
                        mini_asks = orderbook['asks'][j][0] 
        range_asks =   maxi_asks  - mini_asks
        
        print(f"Ask Size : {sum_size_asks}" )
        print(f"Ask range : {range_asks}")

        plt.subplot(2,2,3)
        plt.bar(prices['bids'] , sizes['bids_size'] , color = "green")
        plt.subplot(2,2,4)
        plt.bar(prices['asks'] , sizes['asks_size'] , color = "red")
        time.sleep(2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS
        plt.savefig(os.path.join(plots_directory, f'macd_plot_{timestamp}.png'))  
        return sum_size,rangebid,sum_size_asks,range_asks




filename = "ohlc_data.xlsx"



def append_to_excel(new_data, filename):
    """
    Append new OHLC data to the DataFrame and ensure correct structure,
    then save it into an Excel sheet.
    
    Parameters:
    new_data (dict): The latest OHLC data as a dictionary.
    filename (str): The Excel filename to save the data to.
    """
    # Column names
    columns = ['start_time', 'Open', 'High', 'Low', 'Close', 'volume', 'end_time']
    
    # Convert the new_data to DataFrame (single row)
    new_df = pd.DataFrame([new_data], columns=columns)

    # Try to read the existing Excel file if it exists, otherwise create an empty DataFrame
    try:
        ohlc_df = pd.read_excel(filename)
    except FileNotFoundError:
        ohlc_df = pd.DataFrame(columns=columns)

    # Append the new data to the DataFrame
    ohlc_df = pd.concat([ohlc_df, new_df], ignore_index=True)

    # Ensure the DataFrame has only the last 500 rows
    if len(ohlc_df) > 500:
        ohlc_df = ohlc_df.iloc[-500:]  # Keep the last 500 rows instead of the first 500

    # Write the updated DataFrame to the Excel file
    with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
        ohlc_df.to_excel(writer, index=False, sheet_name='OHLC Data')


# Your WebSocket async function for Binance Spot
async def get_ohlc(symbol="btcusdt", interval="1m"):
    # WebSocket URL for Binance Spot kline (candlestick) stream
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"

    async with websockets.connect(url) as websocket:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connected to {url}")

        while True:
            # Receive and parse the response
            response = await websocket.recv()
            data = json.loads(response)

            # Extract OHLC data from the 'k' field in the response
            kline = data['k']
            is_kline_closed = kline['x']  # Whether this kline is closed

            if is_kline_closed:
                ohlc = {
                    'start_time': kline['t'],  # Kline start time
                    'Open': float(kline['o']),  # Open price
                    'High': float(kline['h']),  # High price
                    'Low': float(kline['l']),  # Low price
                    'Close': float(kline['c']),  # Close price
                    'volume': float(kline['v']),  # Volume
                    'end_time': kline['T']  # Kline end time
                }
                print(ohlc)
                append_to_excel(ohlc, filename)

                # Autocorrelation and MACD calculations
                lag = 1  # Lag period
                length = 12  # Lookback period
                
                df = pd.read_excel(filename)
                print(df)
                df['OHLC4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
                # Step 1: Calculate log returns (log differences)
                df['Log_Return'] = np.log(df['OHLC4']) - np.log(df['OHLC4'].shift(1))

                # Step 2: Calculate the mean of log returns over the specified length
                df['Mean_Log_Return'] = df['Log_Return'].ewm(span=length, adjust=False).mean()

                # Step 3: Calculate deviations from the mean
                df['Dev_Log_Return'] = df['Log_Return'] - df['Mean_Log_Return']

                # Step 4: Calculate autocorrelation on log returns
                autocorr = df['Dev_Log_Return'].rolling(window=length).apply(
                    lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else np.nan
                )

                # Step 5: Calculate confidence intervals (assuming normal distribution)
                z_value = 1.645  # 90% confidence interval
                upper_bound = z_value / np.sqrt(length)
                lower_bound = -z_value / np.sqrt(length)

                # MACD calculation
                fast_length = 3
                slow_length = 12
                signal_length = 3

                # Calculating EMAs (Exponential Moving Averages)
                def ema(series, period):
                    return series.ewm(span=period, adjust=False).mean()

                # Calculating MACD
                df['fast_ma'] = ema(df['OHLC4'], fast_length)
                df['slow_ma'] = ema(df['OHLC4'], slow_length)
                df['MACD'] = df['fast_ma'] - df['slow_ma']
                df['Signal'] = ema(df['MACD'], signal_length)
                df['Histogram'] = df['MACD'] - df['Signal']

                # Plot MACD
                plt.figure(figsize=(12, 6)) 

                plt.cla()
                plt.subplot(2, 2, 1)
                plt.plot(df['MACD'], label='MACD', color='#2962FF', linewidth=1.5)
                plt.plot(df['Signal'], label='Signal', color='#FF6D00', linewidth=1.5)
                colors = np.where(df['Histogram'] >= 0, '#26A69A', '#FF5252')
                plt.bar(df.index, df['Histogram'], color=colors, label='Histogram', width=0.8)
                plt.axhline(0, color='#787B86', linewidth=1.0, linestyle='--', alpha=0.5)
                plt.title('MACD Indicator')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()

                # Plot autocorrelation
                plt.subplot(2, 2, 2)
                plt.plot(df.index, autocorr.ewm(span=5).mean(), color='blue', linewidth=2, label='Autocorrelation on Returns')
                plt.axhline(upper_bound, color='red', linestyle='--', label='Upper Confidence Bound')
                plt.axhline(lower_bound, color='red', linestyle='--', label='Lower Confidence Bound')
                plt.axhline(0, color='gray', linestyle='--', label='Zero Line')
                plt.title('Autocorrelation on Returns')
                plt.xlabel('Time')
                plt.ylabel('Autocorrelation')
                plt.legend()

                bidsize, bidrange, asksize, askrange = orderbook()

                
                if (df['fast_ma'].iloc[-1] >= 30 and (df['fast_ma'].iloc[-1] < df['fast_ma'].iloc[-3])):  # down reversion #ask
                    if (autocorr.ewm(span=5).mean().iloc[-1] > upper_bound):
                        if (df['Close'].iloc[-1] > df['Close'].iloc[-5]):  # up momentum #bid
                            # check orderbook
                            if (bidsize > asksize):
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BUY due to down reversion and up momentum and bidsize > asksize")
                                send_push_notification("BUY", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BUY due to down reversion and up momentum and bidsize > asksize  \n    bidsize: {bidsize}  asksize : {asksize}")
                            else:
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SELL due to down reversion and up momentum and bidsize < asksize")
                                send_push_notification("SELL", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SELL due to down reversion and up momentum and bidsize < asksize  \n    bidsize: {bidsize}  asksize : {asksize}")

                if (df['fast_ma'].iloc[-1] <= -26 and (df['fast_ma'].iloc[-1] > df['fast_ma'].iloc[-3])):  # up reversion #bid
                    if (autocorr.ewm(span=5).mean().iloc[-1] > upper_bound):
                        if (df['Close'].iloc[-1] < df['Close'].iloc[-5]):  # down momentum #ask
                            # check orderbook
                            if (bidsize > asksize):
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BUY due to up reversion and down momentum and bidsize > asksize")
                                send_push_notification("BUY", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BUY due to up reversion and down momentum and bidsize > asksize  \n    bidsize: {bidsize}  asksize : {asksize}")
                            else:
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SELL due to up reversion and down momentum and bidsize < asksize")
                                send_push_notification("SELL", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SELL due to up reversion and down momentum and bidsize < asksize  \n    bidsize: {bidsize}  asksize : {asksize}")

# Running the function using asyncio
asyncio.get_event_loop().run_until_complete(get_ohlc())



              







