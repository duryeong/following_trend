import streamlit as st
import requests
import os
import sys
import subprocess

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    # build
    os.system("./configure --prefix=/home/adminuser/venv/")
    os.system("make")
    # install
    os.system("mkdir -p /home/adminuser/venv/")
    os.system("make install")
    os.system("ls -la /home/adminuser/venv/")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/adminuser/venv/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/adminuser/venv/lib/", "--global-option=-I/home/adminuser/venv/include/", "ta-lib==0.4.24"])
finally:
    import talib as ta

import streamlit as st
import pandas as pd
import yfinance as yf

def make_idx(df, r1=7, ad=14, limad=12, wmean=4 ,iyear=None):
    df[f'rsi{r1}'] = ta.RSI(df['close'], timeperiod=r1)
    df[f'rsi{r1*2}'] = ta.RSI(df['close'], timeperiod=r1*2)
    df[f'rsi{r1*3}'] = ta.RSI(df['close'], timeperiod=r1*3)
    df[f'adx_{ad}'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=ad)
    df[f'mean{wmean}'] = df.close.rolling(window=wmean).mean()

    is_up = []
    for idf in df.iloc:
        # is_up.append(idf.rsi7 > idf.rsi14 > idf.rsi21 and idf.adx > 20)
        is_up.append(idf[f'rsi{r1}'] > idf[f'rsi{r1*2}'] > idf[f'rsi{r1*3}'] and idf[f'adx_{ad}'] > limad and idf.close > idf[f'mean{wmean}'])
    df['is_up'] = is_up
    df['differ'] = (df.close - df.open)/df.open*100
    df = df[['open', 'high', 'low', 'close', 'differ', 'is_up']]
    df = df[::-1]
    return df

def get_stock(c='AAPL'):
    stock_code = c
    # Yahoo Finance에서 주식 정보를 가져옵니다.
    stock_data = yf.Ticker(stock_code)
    df = stock_data.history(interval='1d', period='2mo')
    df.columns = [ic.lower() for ic in list(df.columns)]
    df = df[['open', 'high', 'low', 'close']]
    return df

@st.cache_resource
def get_stock_info():
    df = pd.read_csv('yfinance_anal.csv')
    df = df.sort_values(by='best_value', ascending=False)
    df = df[['tickers', 'best_value', 'best_param']]
    return df.head(100)

stock_info = get_stock_info()
st.title(f'HOME')
st.subheader(f"anal_date:{pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')}")
tabs = st.tabs([f"itab_{inum+1:03d}" for inum, itab in enumerate(stock_info.tickers.values)])

for inum, itab in enumerate(tabs):
    with itab:
        with st.spinner(f'make info {stock_info.tickers.values[inum]}'):
            # st.write(f'{stock_info.tickers.values[inum]}')
            st.write(f'{stock_info.best_value.values[inum]*100:.2f}')
            candle = get_stock(c=stock_info.tickers.values[inum])
            candle = make_idx(candle)
            candle

