import streamlit as st
import pandas as pd
import yfinance as yf
import copy

def ADX(df, period=14):
    data = copy.deepcopy(df)
    # True Range 계산
    data['high-low'] = data['high'] - data['low']
    data['high-close-Prev'] = abs(data['high'] - data['close'].shift(1))
    data['low-close-Prev'] = abs(data['low'] - data['close'].shift(1))
    data['TrueRange'] = data[['high-low', 'high-close-Prev', 'low-close-Prev']].max(axis=1)

    # +DI, -DI 계산
    data['+DM'] = (data['high'] - data['high'].shift(1)).where((data['high'] - data['high'].shift(1)) > 0, 0)
    data['-DM'] = (data['low'].shift(1) - data['low']).where((data['low'].shift(1) - data['low']) > 0, 0)

    data['+DI'] = (data['+DM'].rolling(window=period).sum() / data['TrueRange'].rolling(window=period).sum()) * 100
    data['-DI'] = (data['-DM'].rolling(window=period).sum() / data['TrueRange'].rolling(window=period).sum()) * 100

    # ADX 계산
    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100
    data['ADX'] = data['DX'].rolling(window=period).mean()
    return data['ADX']

def SMA(data, period=30, column='close'):
    return data[column].rolling(window=period).mean()

def RSI(data, period=14, column='close'):
    delta = data[column].diff(1)
    delta = delta.dropna()

    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down

    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))
    RS = AVG_Gain / AVG_Loss

    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI

def make_idx(df, r1=7, ad=14, limad=12, wmean=4 ,iyear=None):
    for itime in range(1,4):
        df[f'rsi{r1*itime}'] = RSI(df, period=r1*itime)
    df[f'adx_{ad}'] = ADX(df, period=ad)
    df[f'mean{wmean}'] = df.close.rolling(window=wmean).mean()

    is_up = []
    for idf in df.iloc:
        # is_up.append(idf.rsi7 > idf.rsi14 > idf.rsi21 and idf.adx > 20)
        is_up.append(idf[f'rsi{r1}'] > idf[f'rsi{r1*2}'] > idf[f'rsi{r1*3}'] and idf[f'adx_{ad}'] > limad and idf.close > idf[f'mean{wmean}'])
    df['is_up'] = is_up
    df['differ'] = (df.close - df.open)/df.open*100
    df = df[['open', 'close', 'differ', 'is_up']]
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
st.title(f'Following Trend')
st.subheader(f"anal_date : ({pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')})")

last_is_up_list = []
last_tab_list = []
with st.expander('sea all_data'):
    with st.spinner(f'make stodk info '):
        tabs = st.tabs([f"{itab}_{inum+1:03d}" for inum, itab in enumerate(stock_info.tickers.values)])
        # with st.expander("all_data", False):
        for inum, itab in enumerate(tabs):
            with itab:
                # st.write(f'{stock_info.tickers.values[inum]}')
                st.write(f'{stock_info.best_value.values[inum]*100:.2f}')
                candle = get_stock(c=stock_info.tickers.values[inum])
                candle = make_idx(candle)
                if candle.is_up.values[0]:
                    last_tab_list.append(itab)
                    last_is_up_list.append(candle)
                st.dataframe(candle, use_container_width=True)

"---"

st.subheader(f"last up list length: {len(last_is_up_list)}")
try:
    is_up_tabs = st.tabs([f"{itab}_{inum + 1:03d}" for inum, itab in enumerate(last_tab_list)])
    for inum, itab in enumerate(is_up_tabs):
        with itab:
            st.dataframe(last_is_up_list[itab], use_container_width=True)
except Exception as e:
    pass

"---"


