import streamlit as st
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import pandas_ta as tb
import pyupbit
import numpy as np
import mplfinance as mpf
from pykrx import stock
import os

def return_name(market):
    Market = []
    for ticker in market:
        Value = stock.get_market_ticker_name(ticker)
        Market.append([Value,ticker])

    df = pd.DataFrame(Market,columns=['회사명','상장번호'])
    return df

def get_kmarket_df():
    KOSDAQ = stock.get_market_ticker_list(market="KOSDAQ")
    KOSPI = stock.get_market_ticker_list(market="KOSPI")
    KOSDAQ.extend(KOSPI)

    df = return_name(KOSDAQ)
    return df

def make_idx(df, r1=7, ad=14, limad=12, wmean=4 ,iyear=None):
    df[f'rsi{r1*1}'] = tb.rsi(df['close'], length=r1*1)
    df[f'rsi{r1*2}'] = tb.rsi(df['close'], length=r1*2)
    df[f'rsi{r1*3}'] = tb.rsi(df['close'], length=r1*3)
    df[f'adx_{ad}'] = tb.adx(df['high'], df['low'], df['close'], length=ad).iloc[:,0]
    df[f'mean{wmean}'] = df.close.rolling(window=wmean).mean()

    is_up = []
    for idf in df.iloc:
        # is_up.append(idf.rsi7 > idf.rsi14 > idf.rsi21 and idf.adx > 20)
        is_up.append(idf[f'rsi{r1}'] > idf[f'rsi{r1*2}'] > idf[f'rsi{r1*3}'] and idf[f'adx_{ad}'] > limad and idf.close > idf[f'mean{wmean}'])
    df['is_up'] = is_up
    df['pre_close'] = df.close.shift(1)
    df['differ'] = (df['close']-df['pre_close'])/df['pre_close']*100

    df = df[['is_up', 'open', 'close', 'differ']]
    df = df[::-1]
    return df

def get_coin(c='KRW-BTC'):
    stock_code = c
    # Yahoo Finance에서 주식 정보를 가져옵니다.
    df = pyupbit.get_ohlcv(c, count=30*6)
    df.columns = [ic.lower() for ic in list(df.columns)]
    df = df[['open', 'high', 'low', 'close']]
    return df

def get_kstock(icoin):
    df = fdr.DataReader(icoin)
    df.index = pd.to_datetime(df.index)
    df.columns = [ic.lower() for ic in list(df.columns)]
    df = df[-60:]
    return df

def get_stock(c='AAPL'):
    stock_code = c
    # Yahoo Finance에서 주식 정보를 가져옵니다.
    stock_data = yf.Ticker(stock_code)
    df = stock_data.history(interval='1d', period='6mo')
    df.columns = [ic.lower() for ic in list(df.columns)]
    df = df[['open', 'high', 'low', 'close']]
    return df

@st.cache_data
def get_stock_info():
    df = pd.read_csv('kstock_anal.csv')
    df = df.sort_values(by='best_value', ascending=False)
    df = df[df.best_value != np.inf]
    df = df[['tickers', 'best_value', 'best_param']]
    return df.head(30)

def ntoname(df, n):
    try:
        return df[df['상장번호'] == n]['회사명'].values[0]
    except:
        return n

def web_main():
    stock_info = get_stock_info()
    with st.spinner(f'make kcoin info '):
        fkstock_info = 'kstock_info.pickle'
        if not os.path.exists(fkstock_info):
            kstock_info = get_kmarket_df()
            kstock_info.to_pickle(fkstock_info)
        else:
            kstock_info = pd.read_pickle(fkstock_info)
    st.title(f'Following Trend kstock')
    st.subheader(f"anal_date : ({(pd.to_datetime('today')+pd.to_timedelta(9, unit='hour')).strftime('%Y-%m-%d %H:%M')})")

    last_is_up_list = []
    last_tab_list = []
    with st.expander('sea all_data'):
        with st.spinner(f'make coin info '):
            tabs = st.tabs([f"{ntoname(kstock_info,itab)}_{inum+1:03d}" for inum, itab in enumerate(stock_info.tickers.values)])
            # with st.expander("all_data", False):
            for inum, itab in enumerate(tabs):
                with itab:
                    # st.write(f'{stock_info.tickers.values[inum]}')
                    st.write(f'{stock_info.best_value.values[inum]*100:.2f}')
                    candle = get_kstock(stock_info.tickers.values[inum])
                    candle = make_idx(candle)
                    if candle.is_up.values[0]:
                        last_tab_list.append(stock_info.tickers.values[inum])
                        last_is_up_list.append(candle)
                    st.dataframe(candle, use_container_width=True)

    "---"

    st.subheader(f"last up list length: {len(last_is_up_list)}")
    try:
        is_up_tabs = st.tabs([f"{ntoname(kstock_info,itab)}_{inum + 1:03d}" for inum, itab in enumerate(last_tab_list)])
        for inum, itab in enumerate(is_up_tabs):
            with itab:
                st.dataframe(last_is_up_list[inum], use_container_width=True)
                fig_df = get_kstock(last_tab_list[inum])
                st.subheader(f"{ntoname(kstock_info, last_tab_list[inum])}")
                fig, ax = mpf.plot(fig_df, style='default', type='candle', returnfig=True)
                st.pyplot(fig)
    except Exception as e:
        pass

    "---"


if __name__ == "__main__":
    web_main()
