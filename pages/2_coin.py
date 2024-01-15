import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as tb
import pyupbit
import copy

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
    df['differ'] = (df['close']-df['open'])/df['open']*100

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
    df = pd.read_csv('coin_anal.csv')
    df = df.sort_values(by='best_value', ascending=False)
    df = df[['tickers', 'best_value', 'best_param']]
    return df.head(100)

def web_main():
    stock_info = get_stock_info()
    st.title(f'Following Trend')
    st.subheader(f"anal_date : ({(pd.to_datetime('today')+pd.to_timedelta(9, unit='hour')).strftime('%Y-%m-%d %H:%M')})")

    last_is_up_list = []
    last_tab_list = []
    with st.expander('sea all_data'):
        with st.spinner(f'make coin info '):
            tabs = st.tabs([f"{itab}_{inum+1:03d}" for inum, itab in enumerate(stock_info.tickers.values)])
            # with st.expander("all_data", False):
            for inum, itab in enumerate(tabs):
                with itab:
                    # st.write(f'{stock_info.tickers.values[inum]}')
                    st.write(f'{stock_info.best_value.values[inum]*100:.2f}')
                    candle = get_coin(c=stock_info.tickers.values[inum])
                    candle = make_idx(candle)
                    if candle.is_up.values[0]:
                        last_tab_list.append(stock_info.tickers.values[inum])
                        last_is_up_list.append(candle)
                    st.dataframe(candle, use_container_width=True)

    "---"

    st.subheader(f"last up list length: {len(last_is_up_list)}")
    try:
        is_up_tabs = st.tabs([f"{itab}_{inum + 1:03d}" for inum, itab in enumerate(last_tab_list)])
        for inum, itab in enumerate(is_up_tabs):
            with itab:
                st.dataframe(last_is_up_list[inum], use_container_width=True)
    except Exception as e:
        pass

    "---"


if __name__ == "__main__":
    web_main()
