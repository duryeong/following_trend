import os
import pandas as pd
import yfinance as yf
import pandas_ta as tb
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import ccxt  # ccxt 라이브러리 추가
import time
import concurrent.futures  # 병렬 처리를 위한 모듈 추가
import streamlit as st  # streamlit 라이브러리 추가

top = 1
total_m = 1000

def make_idx(df, r1=7, ad=14, limad=12, mc_ratio=100.01, wmean=4 ,iyear=None):
    df[f'rsi{r1}'] = tb.rsi(df['close'], length=r1)
    df[f'rsi{r1*2}'] = tb.rsi(df['close'], length=r1*2)
    df[f'rsi{r1*3}'] = tb.rsi(df['close'], length=r1*3)
    df[f'adx_{ad}'] = tb.adx(df['high'], df['low'], df['close'], length=ad).iloc[:,0]
    df[f'mean{wmean}'] = df.close.rolling(window=wmean).mean()
    df[f'mc_ratio_{mc_ratio}'] = df.close / df[f'mean{wmean}']

    is_up = []
    for idf in df.iloc:
        # is_up.append(idf.rsi7 > idf.rsi14 > idf.rsi21 and idf.adx > 20)
        is_up.append(idf[f'rsi{r1}'] > idf[f'rsi{r1*2}'] > idf[f'rsi{r1*3}'] and idf[f'adx_{ad}'] > limad and idf[f'mc_ratio_{mc_ratio}'] < mc_ratio and idf.close > idf[f'mean{wmean}'])
    df['is_up'] = is_up
    df['real_is_up'] = df.is_up.shift(1)

    df['pre_close'] = df.close.shift(1)
    df['differ'] = (df['close']-df['pre_close'])/df['pre_close']*100

    return df

def load_upbit_info():
    df = pd.read_csv('coin_anal_upbit_for_optimization_2022_2023.csv')
    eth_df = df[df.tickers == 'KRW-ETH']
    btc_df = df[df.tickers == 'KRW-BTC']
    n_df = df[df.tickers != 'KRW-ETH']
    n_df = n_df[n_df.tickers != 'KRW-BTC']
    n_df = n_df.sort_values(by='best_value', ascending=False)

    return pd.concat([btc_df, n_df])

def update_upbit():
    upbit_info = load_upbit_info()
    upbit_info = upbit_info.head(top)
    upbit_dict = {}  # 딕셔너리 초기화

    def fetch_data(ticker):
        print(ticker)
        upbit = ccxt.upbit()  # upbit 인스턴스 생성
        start_date = pd.Timestamp('2024-01-01')
        end_date = pd.Timestamp('today')
        current_date = start_date

        # DataFrame 초기화
        upbit_dict[ticker] = pd.DataFrame()  # ticker에 대한 DataFrame 초기화

        while current_date <= end_date:
            df = upbit.fetch_ohlcv(ticker, timeframe='1d', since=int(current_date.timestamp() * 1000), limit=200)  # OHLCV 데이터 가져오기
            df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # 데이터프레임으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 타임스탬프 변환
            df.set_index('timestamp', inplace=True)  # 타임스탬프를 인덱스로 설정
            df.columns = [ic.lower() for ic in list(df.columns)]  # 열 이름 소문자로 변환
            
            # 데이터가 있을 경우에만 DataFrame에 추가
            if not df.empty:
                upbit_dict[ticker] = pd.concat([upbit_dict[ticker], df])  # DataFrame에 데이터 추가
                current_date = df.index[-1] + timedelta(days=1)  # 마지막 데이터의 날짜에서 1일 후로 설정
            else:
                print(f"{ticker}에 대한 데이터가 없습니다. 날짜: {current_date}")
                current_date += timedelta(days=1)  # 데이터가 없더라도 날짜 증가

        # 데이터 가져오기 완료 후 마지막 데이터 출력

    # 병렬 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(fetch_data, upbit_info['tickers'])

    return upbit_dict  # 딕셔너리 반환

def main(upbit_dict):  # upbit_dict 매개변수 추가
    dates = []
    returns = []
    keeps = []
    upbit_info = load_upbit_info()
    upbit_info = upbit_info.head(top)
    for ikey in upbit_dict.keys():
        params = eval(upbit_info[upbit_info['tickers'] == ikey]['best_param'].values[0])
        upbit_dict[ikey] = make_idx(upbit_dict[ikey], r1=params['r1'], ad=params['ad'], limad=params['limad'], mc_ratio=params['mc_ratio'], wmean=params['wmean'])

    for idate in pd.date_range((pd.to_datetime('today')-pd.to_timedelta(365, unit='D')).strftime("%Y-%m-%d"), pd.to_datetime('today'), freq='D'):
        print(idate)
        indate = pd.to_datetime(idate.strftime('%Y%m%d'))
        dates.append(indate)
        re_sum = 0
        re_count = 0
        coins = []
        for ikey in upbit_dict.keys():
            rec = upbit_dict[ikey][upbit_dict[ikey].index == idate]
            if rec.real_is_up.values[0]:
                re_sum = re_sum + rec.differ.values[0]
                re_count = re_count + 1
                coins.append(ikey)
        if re_count == 0:
            re = np.nan
        else:
            re = re_sum/re_count
        returns.append(re)
        keeps.append("/".join(coins))

    df = pd.DataFrame()
    df['date'] = dates
    df['returns'] = np.array(returns)/100 + 1
    df = df.fillna(1)
    df['sum_returns'] = df.returns.cumprod()*100
    df['keeps'] = np.array(keeps)
    # df.to_csv('table_upbit.csv')
    return df

def web_main(df):
    if st.button('rerun'):
        st.rerun()
    st.title(f'Following Trend: BTC')
    df.returns = (df.returns - 1)*100
    df = df.round(2)
    df.date = df.date.map(lambda x: x.strftime("%Y-%m-%d"))
    st.dataframe(df[::-1][["date","keeps", "returns", "sum_returns"]], use_container_width=True)



def any_empty_dict(indict):
    for ikey in indict.keys():
        if len(indict[ikey]) == 0:
            return True
    return False

if __name__ == "__main__":
    start = time.time()
    upbit_dict = update_upbit()  # 딕셔너리에 데이터 저장
    while any_empty_dict(upbit_dict):
        upbit_dict = update_upbit()  # 딕셔너리에 데이터 저장
    print(time.time() - start)
    df = main(upbit_dict)  # 딕셔너리 전달 
    print(any_empty_dict(upbit_dict))
    web_main(df)