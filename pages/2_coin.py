import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as tb
import pyupbit
import mplfinance as mpf

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

def check_buy(c):
    if c.is_up.values[0] and not c.is_up.values[1]:
        return True
    return False

def check_sell(c):
    if c.is_up.values[1] and not c.is_up.values[0]:
        return True
    return False

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
    eth_df = df[df.tickers == 'KRW-ETH']
    neth_df = df[df.tickers != 'KRW-ETH']
    odf = pd.concat([eth_df, neth_df.head(50)])
    return odf

def web_main():
    stock_info = get_stock_info()
    st.title(f'Following Trend')
    st.subheader(f"anal_date : ({(pd.to_datetime('today')+pd.to_timedelta(9, unit='h')).strftime('%Y-%m-%d %H:%M')})")

    buy_list = []
    sell_list = []
    last_is_up_list = []
    last_tab_list = []
    with st.expander('see all_data'):
        with st.spinner(f'make coin info '):
            tabs = st.tabs([f"{itab}_{inum+1:03d}" for inum, itab in enumerate(stock_info.tickers.values)])
            # with st.expander("all_data", False):
            for inum, itab in enumerate(tabs):
                with itab:
                    # st.write(f'{stock_info.tickers.values[inum]}')
                    st.write(f'{stock_info.best_value.values[inum]*100:.2f}')
                    candle = get_coin(c=stock_info.tickers.values[inum])
                    info = eval(stock_info['best_param'].values[inum])
                    candle = make_idx(candle, info['r1'], info['ad'], info['limad'], info['wmean'])
                    t = stock_info.tickers.values[inum]
                    if check_buy(candle):buy_list.append(t)
                    if check_sell(candle):sell_list.append(t)
                    if candle.is_up.values[0]:
                        last_tab_list.append(t)
                        last_is_up_list.append(candle)
                    st.dataframe(candle, use_container_width=True)

    "---"


    with st.expander('see 보유_data'):
        st.subheader(f"last up list length: {len(last_is_up_list)}")
        try:
            is_up_tabs = st.tabs([f"{itab}_{inum + 1:03d}" for inum, itab in enumerate(last_tab_list)])
            for inum, itab in enumerate(is_up_tabs):
                with itab:
                    st.dataframe(last_is_up_list[inum], use_container_width=True)
                    fig_df = get_coin(last_tab_list[inum])
                    fig, ax = mpf.plot(fig_df, style='default', type='candle', title=f"{last_tab_list[inum]}", returnfig=True)
                    st.pyplot(fig)
        except Exception as e:
            pass

    "---"
    st.subheader('buy_list')
    try:
        st.write(", ".join(buy_list))
    except Exception as e:
        pass
    "---"
    st.subheader('sell_list')
    try:
        st.write(", ".join(sell_list))
    except Exception as e:
        pass

if __name__ == "__main__":
    st.set_page_config(page_title='following trend')
    web_main()
