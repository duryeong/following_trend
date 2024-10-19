import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as tb
import numpy as np
import copy
import ccxt


def get_ohlcv_binance(symbol):
    # Binance 거래소 객체 생성
    exchange = ccxt.binanceus()

    # 심볼과 타임프레임 설정
    timeframe = '1d'  # 타임프레임 (1일)

    # 캔들스틱 데이터 불러오기
    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=5*365)

    # 데이터 프레임으로 변환
    columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(candles, columns=columns)

    # 타임스탬프를 날짜 형식으로 변환
    df['Timestamp'] = pd.to_datetime(pd.to_datetime(df['Timestamp'], unit='ms'))

    # 데이터 출력
    return df

def calculate_indicators(df, r1, ad, wmean):
    df[f'rsi{r1}'] = tb.rsi(df['close'], length=r1)
    df[f'rsi{r1 * 2}'] = tb.rsi(df['close'], length=r1 * 2)
    df[f'rsi{r1 * 3}'] = tb.rsi(df['close'], length=r1 * 3)
    df[f'adx_{ad}'] = tb.adx(df['high'], df['low'], df['close'], length=ad).iloc[:, 0]
    df[f'mean{wmean}'] = df.close.rolling(window=wmean).mean()

def make_idx_rev(idf, r1=7, ad=14, limad=12, wmean=4):
    df = copy.deepcopy(idf)
    calculate_indicators(df, r1, ad, wmean)
    df['is_short'] = df.apply(lambda row: row[f'rsi{r1}'] < row[f'rsi{r1 * 2}'] < row[f'rsi{r1 * 3}']
        and row[f'adx_{ad}'] > limad
        and row.close < row[f'mean{wmean}'], axis=1)

    return df


def make_idx(idf, r1=7, ad=14, limad=12, wmean=4 ,iyear=None):
    df = copy.deepcopy(idf)
    calculate_indicators(df, r1, ad, wmean)

    df['is_long'] = df.apply(lambda row: row[f'rsi{r1}'] > row[f'rsi{r1 * 2}'] > row[f'rsi{r1 * 3}']
       and row[f'adx_{ad}'] > limad
       and row.close > row[f'mean{wmean}'], axis=1)

    return df

def check_buy(c):
    if c.is_long.values[1] and not c.is_long.values[2]:
        return True
    return False

def check_sell(c):
    if c.is_long.values[2] and not c.is_long.values[1]:
        return True
    return False

def get_coin(c='KRW-BTC'):
    stock_code = c
    # Yahoo Finance에서 주식 정보를 가져옵니다.
    binance_c = f"{c.split('-')[1]}/USDT"
    df = get_ohlcv_binance(binance_c)
    # print(df)
    df.columns = [ic.lower() for ic in list(df.columns)]
    df.index = df.timestamp
    return df

def get_stock(c='AAPL'):
    stock_code = c
    # Yahoo Finance에서 주식 정보를 가져옵니다.
    stock_data = yf.Ticker(stock_code)
    df = stock_data.history(interval='1d', period='6mo')
    df.columns = [ic.lower() for ic in list(df.columns)]
    df = df[['open', 'high', 'low', 'close']]
    return df


def get_short_info():
    df = pd.read_csv('coin_anal_short_binance.csv')
    df = df.sort_values(by='best_value', ascending=False)
    df = df[['tickers', 'best_value', 'best_param']]
    return df

def get_long_info():
    df = pd.read_csv('coin_anal_binance.csv')
    df = df.sort_values(by='best_value', ascending=False)
    df = df[['tickers', 'best_value', 'best_param']]
    return df

#@st.cache_data
def get_stock_info():
    df = pd.read_csv('coin_anal.csv')
    df = df.sort_values(by='best_value', ascending=False)
    df = df[['tickers', 'best_value', 'best_param']]
    eth_df = df[df.tickers == 'KRW-ETH']
    btc_df = df[df.tickers == 'KRW-BTC']
    neth_df = df[df.tickers != 'KRW-ETH']
    odf = pd.concat([eth_df, btc_df, neth_df.head(18)])
    return odf

def get_profit(candle):
    import copy
    df = copy.deepcopy(candle[::-1])
    df['pre_up'] = df.is_long.shift(1)
    df['pre_close'] = df.close.shift(1)

    df['ror'] = np.where(df['pre_up'],
                         df['close'] / df['pre_close'],
                         1)
    df['hpr'] = df['ror'].cumprod()

    # Draw Down 계산 (누적 최대 값과 현재 hpr 차이 / 누적 최대값 * 100)
    df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
    # print(pd.to_datetime('today'))
    # print(df)
    # print(pd.to_datetime('today'))
    # print(df)
    return df

def get_profit_short(candle):
    import copy
    df = copy.deepcopy(candle[::-1])
    df['pre_down'] = df.is_short.shift(1)
    df['pre_close'] = df.close.shift(1)

    df['ror'] = np.where(df['pre_down'],
                         df['pre_close'] / df['close'],
                         1)
    df['hpr'] = df['ror'].cumprod()

    # Draw Down 계산 (누적 최대 값과 현재 hpr 차이 / 누적 최대값 * 100)
    df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
    # print(pd.to_datetime('today'))
    # print(df)
    # print(pd.to_datetime('today'))
    # print(df)
    return df


def calculate_mdd_for_each_position(df):
    mdd_list = []  # 각 포지션의 MDD를 저장할 리스트
    df['timestamp'] = df.index

    peak = None
    mdd = 0
    is_in_position = False
    start_position_index = None

    for i, row in df.iterrows():
        if row['O']:  # O 컬럼이 True일 때만 long 포지션으로 간주
            if not is_in_position:
                # 포지션 시작 (long 포지션 진입)
                is_in_position = True
                peak = row['close']  # 첫 진입 시의 가격을 peak으로 설정
                start_position_index = i  # 포지션 시작 인덱스
            else:
                # 포지션이 활성화된 상태에서 peak과 drawdown 계산
                peak = max(peak, row['close'])
                drawdown = (peak - row['close']) / peak
                mdd = max(mdd, drawdown)
        else:
            if is_in_position:
                # 포지션 종료 시 해당 포지션의 MDD를 기록
                mdd_list.append({
                    'start': df.loc[start_position_index, 'timestamp'],
                    'end': row['timestamp'],
                    'mdd': mdd
                })

                # 포지션 초기화
                is_in_position = False
                mdd = 0
                peak = None

    # 마지막 포지션의 MDD 처리 (포지션이 끝나지 않았을 경우)
    if is_in_position and start_position_index is not None:
        mdd_list.append({
            'start': df.loc[start_position_index, 'timestamp'],
            'end': df.iloc[-1]['timestamp'],  # 마지막 날짜
            'mdd': mdd
        })

    df = pd.DataFrame(mdd_list)
    return mdd_list, df.mdd.max()*100

def web_main():
    if st.button('rerun'):
        st.rerun()
    # stock_info = get_stock_info()
    long_info = get_long_info()
    short_info = get_short_info()
    st.title(f'Following Trend')
    st.subheader(f"anal_date : ({(pd.to_datetime('today')+pd.to_timedelta(9, unit='h')).strftime('%Y-%m-%d %H:%M')})")

    buy_list = []
    sell_list = []
    last_is_long_list = []
    last_tab_list = []
    with st.expander('see all_data', expanded=True):
        with st.spinner(f'make coin info '):
            tabs = st.tabs([f"{itab}_{inum+1:03d}" for inum, itab in enumerate(long_info.tickers.values)])
            # with st.expander("all_data", False):
            for inum, itab in enumerate(tabs):
                with itab:
                    # st.write(f'{stock_info.tickers.values[inum]}')
                    # st.write(f'{stock_info.best_value.values[inum]*100:.2f}')
                    candle = get_coin(c=long_info.tickers.values[inum])
                    info = eval(long_info['best_param'].values[inum])
                    long_candle = make_idx(candle, info['r1'], info['ad'], info['limad'], info['wmean'])
                    long_dump_df = get_profit(long_candle)
                    info = eval(short_info['best_param'].values[inum])
                    short_candle = make_idx_rev(candle, info['r1'], info['ad'], info['limad'], info['wmean'])
                    short_dump_df = get_profit_short(short_candle)
                    long_candle['is_short'] = short_candle['is_short']
                    long_candle['pre_close'] = long_candle.close.shift(1)
                    long_candle['differ'] = (long_candle.close/long_candle.pre_close-1)*100
                    long_candle = long_candle[['is_long', 'is_short', 'close', 'differ']]
                    long_candle['is_long'][long_candle['is_long'] == True] = 'O'
                    long_candle['is_long'][long_candle['is_long'] == False] = ''
                    long_candle['is_short'][long_candle['is_short'] == True] = 'O'
                    long_candle['is_short'][long_candle['is_short'] == False] = ''
                    long_candle['O'] = long_candle['is_long'].shift(1)
                    mdd_results, max_mdd = calculate_mdd_for_each_position(long_candle)
                    # 결과 출력
                    # for result in mdd_results:
                    #     print(f"포지션 시작: {result['start']}, 포지션 종료: {result['end']}, MDD: {result['mdd'] * 100:.2f}%")
                    st.subheader(f"max MDD: {max_mdd:.1f}%")
                    st.table(long_candle[::-1][['is_long', 'close', 'differ']])
                    st.subheader(f"Long Profit in the last year: {long_dump_df.hpr.values[-1]*100:.2f}%, MDD: {long_dump_df.dd.max():.2f}%")
                    st.subheader(f"Short Profit in the last year: {short_dump_df.hpr.values[-1]*100:.2f}%, MDD: {short_dump_df.dd.max():.2f}%")

                import gc
                gc.collect()

if __name__ == "__main__":
    st.set_page_config(page_title='following trend')
    web_main()
