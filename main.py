# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime , timedelta
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# !pip install -U keras-tuner
from kerastuner.tuners import RandomSearch
from transformers import pipeline


# Functions
def get_dates():
    """ Get the start and end dates for fetching historical stock data. """
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * 5)
    return start_date, end_date

def download_data(stock_name, start_date, end_date):
    """ Download historical stock market data for a given stock within a date range. """
   
    stock = yf.Ticker(stock_name)
    return stock.history(start=start_date, end=end_date)

def modify_dataframe(stock):
    """ Modifies a DataFrame by removing columns, extracting date, and resetting index. """

    stock = stock[['Open','Close']]
    stock['Date'] = stock.index
    stock.Date = stock.Date.dt.date  # Assuming Date is a datetime column
    stock.reset_index(drop=True, inplace=True)
    return stock

def modify_stock(stock):
    """ Modifies a DataFrame by removing columns, extracting date, and resetting index. """

    stock = stock[['Open','Close']]
    stock.reset_index(drop=True, inplace=True)
    return stock

def fill_null_values(stock):
    """ Fills missing (NaN) values in a stock DataFrame using forward fill (ffill) followed by backward fill (bfill)."""
    
    # Check for null values
    if stock.isnull().values.any():
        # Forward fill 
        stock = stock.fillna(method='ffill')
        # Backward fill
        stock = stock.fillna(method='bfill')
    
    return stock

def fetch_and_combine_data(stock, start_date, time_now):
    """ Fetches news articles for a given stock within a specified date range and combines them into a single DataFrame. """
    
    api_url = 'https://api.newscatcherapi.com/v2/search?lang=en&q=tesla'
    api_key = '################'
    headers = {'x-api-key': api_key}
    dfs = []  # List to store DataFrames

    while start_date < time_now:
        end_date = min(start_date + timedelta(days=10), time_now)  # Ensure end date doesn't exceed today

        # Construct URL with date range for the current request
        url_with_dates = f"{api_url}&from={start_date.strftime('%Y/%m/%d')}&to={end_date.strftime('%Y/%m/%d')}"

        response = requests.get(url_with_dates, headers=headers)

        if response.status_code == 200:
            data = response.json()

            if 'articles' in data:
                articles = data['articles']

                if articles:
                    df = pd.DataFrame(articles)
                    dfs.append(df)  # Append DataFrame to the list
                else:
                    print("No articles found in the response data.")
            else:
                print("The response data does not contain an 'articles' key.")
        else:
            print(f"API request failed with status code: {response.status_code}")

        start_date = end_date + timedelta(days=1)  # Move to the next 10-day period

    if dfs:
        if all(df.columns.tolist() == dfs[0].columns.tolist() for df in dfs):
            return pd.concat(dfs, ignore_index=True)
        else:
            print("DataFrames do not have the same columns.")
            return None
    else:
        print("No data collected.")
        return None

def modify_news(news):
    """ Modifies the news DataFrame by selecting relevant columns and formatting the date. """
    
    news = news[['title','published_date']]
    news = news.rename(columns={'published_date': 'Date'})
    news['Date'] = pd.to_datetime(news['Date'])
    news['Date'] = news.Date.dt.date
    return news

def stock_news(stock):
    """ Fetches, processes, and returns news articles for a given stock. """
    news = fetch_and_combine_data(stock, start_date, time_now)
    news=modify_news(news)
    return news



def analysis_news_using_transformer(news):
    """ Analyzes the sentiment of news headlines using a transformer-based model. """
    
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model)
    sent_results = {}

    for i, d in news.iterrows():
        sent = sentiment_task(d["title"])
        sent_results[d["Date"]] = sent

    news_analysis_as_df = pd.DataFrame(sent_results).T
    return news_analysis_as_df


def modify_score(dataframe):
    """ Adjusts sentiment scores for negative and positive labels. """
    
    for index, row in dataframe.iterrows():
        if row['label'] == 'negative':
            dataframe.loc[index, 'score'] = row['score'] - 1
        elif row['label'] == 'positive':
            dataframe.loc[index, 'score'] = row['score'] + 1
    return dataframe


def modify_news_analysis(news):
    """ Processes the sentiment analysis results by extracting and modifying relevant information. """
    
    news["label"] = news[0].apply(lambda x: x["label"])
    news["score"] = news[0].apply(lambda x: x["score"])
    news = news[['label', 'score']]

    news = modify_score(news)
    news.drop('label', axis=1, inplace=True)
    news = news.reset_index(drop=False)
    news = news.rename(columns={'index': 'Date'})
    return news

def analysis_news(news):
    """ Conducts sentiment analysis on news articles and returns the processed results. """
    
    news = analysis_news_using_transformer(news)
    news = modify_news_analysis(news)
    return news

# [[[o1,c1,s1],[o2,c2,s2],[o3,c3,s3],[o4,c4,s4],[o5,c5,s5]]],[o6,c6]
# [[[[o2,c2,s2],[o3,c3,s3],[o4,c4,s4],[o5,c5,s5],[o6,c6,s6]]],[o7,c7]
# [[[[o3,c3,s3],[o4,c4,s4],[o5,c5,s5],[o6,c6,s6],[o7,c7,s7]]],[o8,c8]

def df_to_x_y(df, window_size=5):
    """ Converts a DataFrame or NumPy array into sequences for time series forecasting."""
    
    if isinstance(df, pd.DataFrame):
        df_as_np = df.to_numpy()
    elif isinstance(df_scaled, np.ndarray):
        df_as_np = df
    else:
        raise ValueError("Input must be a pandas DataFrame or numpy array.")

    x = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = df_as_np[i:i+window_size]
        x.append(row)
        label = [df_as_np[i+window_size, 0], df_as_np[i+window_size, 1]]
        y.append(label)
    return np.array(x), np.array(y)

def preprocess(x):
    """ Normalizes the input features using precomputed mean and standard deviation values. """
    x[:,:]= (x[:,:] - open_training_mean) / open_training_std
    x[:,:, 1] = (x[:, :, 1] - close_training_mean) / close_training_std
    x[:,:, 1] = (x[:, :, 2] - score_training_mean) / score_training_std


def preprocess_output(y):
    """ Normalizes the output labels using precomputed mean and standard deviation values. """
    y[:,0]= (y[:,0] - open_training_mean) / open_training_std
    y[:, 1] = (y[:, 1] - close_training_mean) / close_training_std



def main():
    # Scrape historical data
    stock_name = "TSLA"
    start_date, end_date = get_dates()
    stock = download_data(stock_name, start_date, end_date)
    df = modify_stock(stock)
    df = fill_null_values(df)

    # Dealing with news and sentiment analysis
    news = stock_news(stock_name)
    news_analysis = analysis_news(news)

    # Merge historical data and news_analysis
    df = pd.merge(stock,sent_df,how='inner',on='Date')
    df = df.set_index('Date')
    df = df.sort_values('Date')
    df = fill_null_values(df)

    # Prepare data to LSTM
    x,y = df_to_x_y(df)

    length_of_data = x.shape[0]
    x_train = x[:int(length_of_data)*.60 ]
    y_train = y[:int(length_of_data)*.60]
    x_val = x[int(length_of_data)*.60 :int(length_of_data)*.80]
    y_val = y[int(length_of_data)*.60 :int(length_of_data)*.80]
    x_test = x[int(length_of_data)*.80:]
    y_test = y[int(length_of_data)*.80:]

    open_training_mean = np.mean(x_train[:, :])
    open_training_std = np.std (x_train[:, :])
    close_training_mean = np.mean (x_train[:, :, 1])
    close_training_std = np.std (x_train[:,:, 1])
    score_training_mean = np.mean (x_train[:, :, 2])
    score_training_std = np.std (x_train[:,:, 2])

    preprocess(x_train)
    preprocess(x_val)
    preprocess(x_test)

    preprocess_output(y_train)
    preprocess_output(y_val)
    preprocess_output(y_test)

    def build_model(hp):
        model = Sequential()
        model.add(InputLayer((5, 3)))
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(LSTM(units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32), return_sequences=True))
        model.add(LSTM(units=hp.Int('lstm_units_last', min_value=32, max_value=256, step=32), return_sequences=False))
        model.add(Dense(hp.Int('dense_units', min_value=32, max_value=256, step=32), 'relu'))
        model.add(Dense(2, 'linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), metrics=[RootMeanSquaredError()])
        return model


    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='hyperparameter_tuning',
        project_name='lstm_regression'
    )


    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=20, callbacks=[EarlyStopping(patience=3)])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, callbacks=[EarlyStopping(patience=3)])



if __name__ == "__main__":
    main()