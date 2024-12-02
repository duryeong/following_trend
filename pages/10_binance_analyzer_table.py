import os
import streamlit as st
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

def make_idx(df, r1=7, ad=14, limad=12, mc_ratio=1.01, wmean=4 ,iyear=None):
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

    df['pre_close'] = df.close.shift(1)
    df['differ'] = (df['close']-df['pre_close'])/df['pre_close']*100

    return df

def get_binance_recommendations(selected_date=None, all_data=None):
    try:
        binance_info = load_binance_info()
        binance_info = binance_info.head(30)
        
        recommended_binances = []
        
        for idx, (_, row) in enumerate(binance_info.iterrows()):
            ticker = row['tickers']
            if row['best_param'] is None: continue
            try:
                df = all_data[ticker]
                params = eval(row['best_param'])
                df = make_idx(df, r1=params['r1'], ad=params['ad'], limad=params['limad'], mc_ratio=params['mc_ratio'], wmean=params['wmean'])

                if df is None or df.empty:
                    continue
                
                # 날짜 처리 수정
                if selected_date:
                    target_date = pd.to_datetime(selected_date) + pd.Timedelta(9, unit='h')
                    df.index = pd.to_datetime(df.index)
                    
                    # 선택된 날짜 이하의 데이터만 필터링
                    valid_data = df[df.index <= target_date]
                    
                    if valid_data.empty:
                        print(f"{ticker}의 유효한 데이터가 없습니다.")
                        continue
                    
                    last_date = valid_data.index[-1]
                    pre_last_date = valid_data.index[-2]
                    if valid_data.loc[last_date, 'is_up'] or valid_data.loc[pre_last_date, 'is_up']:
                        # 현재 상승 구간의 시작점 찾기
                        period_start = None
                        false_indices = np.where(~valid_data['is_up'])[0]
                        last_false_idx = -1
                        if len(false_indices) > 0:
                            last_false_idx = false_indices[-1] + 1
                            if false_indices[-1] == len(valid_data) - 1:
                                last_false_idx = false_indices[-2] + 1
                        period_start = valid_data.index[last_false_idx]
                        
                        # 수익계산
                        start_price = valid_data.loc[period_start, 'close']
                        end_price = valid_data.loc[last_date, 'close']
                        returns = ((end_price - start_price) / start_price) * 100
                        
                        recommended_binances.append({
                            'ticker': ticker,
                            'close': valid_data.loc[last_date, 'close'],
                            'date': last_date.date(),
                            'is_up': valid_data.loc[last_date, 'is_up'],
                            'returns': round(returns, 2)
                        })
                else:
                    # 최신 데이터 처리도 동일한 방식으로 수정
                    if df['is_up'].values[-1] or df['is_up'].values[-2]:
                        current_period = df
                        period_start = None
                        false_indices = np.where(~current_period['is_up'])[0]
                        last_false_idx = -1
                        if len(false_indices) > 0:
                            last_false_idx = false_indices[-1] + 1
                            if false_indices[-1] == len(current_period) - 1:
                                last_false_idx = false_indices[-2] + 1
                        period_start = current_period.index[last_false_idx]
                            
                        start_price = df.loc[period_start, 'close']
                        end_price = df.loc[target_date, 'close']
                        returns = ((end_price - start_price) / start_price) * 100
                        
                        recommended_binances.append({
                            'ticker': ticker,
                            'close': df.loc[target_date, 'close'],
                            'date': target_date.date(),
                            'is_up': df.loc[target_date, 'is_up'],
                            'returns': round(returns, 2)
                        })
                
            except Exception as e:
                continue
        
        return recommended_binances
        
    except Exception as e:
        print(f"상세 오류 정보:\n{type(e).__name__}: {str(e)}")
        return []

def daily_returns(idate, all_data):  # all_data 파라미터 추가
    recommended_binances = get_binance_recommendations(idate, all_data)  # all_data를 전달
    # 결과를 데이터프레임으로 변환
    if not recommended_binances: return np.nan
    df_results = pd.DataFrame(recommended_binances)
    df_results['close'] = df_results['close'].round(2)
    df_results.columns = ['종목코드', '종가', '기준일', 'is_up', '수익(W)']
    df_results.set_index('종목코드', inplace=True)
    df_results = df_results[['종가', '기준일', 'is_up', '수익(W)']]
    
    # 평균 수익 계산
    if len(df_results[df_results.is_up == False]) > 0:
        average_returns = df_results[df_results.is_up == False]['수익(W)'].sum()
    else:
        average_returns = np.nan
    return average_returns


def load_binance_info():
    return pd.read_csv('coin_anal_binance_for_optimization_241126.csv')

def update_binance():
    binance_info = load_binance_info()
    binance_info = binance_info.head(30)
    all_data = {}  # 데이터를 저장할 딕셔너리 추가

    def fetch_data(ticker):
        print(ticker)
        binance = ccxt.binance()  # Binance 인스턴스 생성
        retries = 100  # 최대 재시도 횟수
        for attempt in range(retries):
            try:
                df = binance.fetch_ohlcv(ticker, timeframe='1d')  # OHLCV 데이터 가져오기
                df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # 데이터프레임으로 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 타임스탬프 변환
                df.set_index('timestamp', inplace=True)  # 타임스탬프를 인덱스로 설정
                df.columns = [ic.lower() for ic in list(df.columns)]  # 열 이름 소문자로 변환
                all_data[ticker] = df  # 딕셔너리에 데이터 저장
                break  # 성공적으로 데이터를 가져오면 루프 종료
            except ccxt.RequestTimeout as e:
                print(f"요청 시간 초과: {e}. 재시도 중... ({attempt + 1}/{retries})")
                if attempt == retries - 1:
                    print(f"{ticker}의 데이터를 가져오는 데 실패했습니다.")
            except Exception as e:
                print(f"오류 발생: {e}")
                break  # 다른 오류 발생 시 루프 종료

    progress_bar = st.progress(0)
    # 병렬 처리
    total_tickers = len(binance_info['tickers'])  # 총 티커 수
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(fetch_data, ticker): ticker for ticker in binance_info['tickers']}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()  # 결과를 기다림
            progress_bar.progress((i+1)/total_tickers)
            # 진행 상황을 출력하거나 업데이트할 수 있는 코드 추가 가능
            # 예: print(f"{futures[future]} 처리 완료")
    
    return all_data  # 딕셔너리 반환

def main(all_data):  # all_data 파라미터 추가
    returns = []
    dates = pd.date_range(pd.to_datetime('today') - pd.to_timedelta(30, unit='D'), pd.to_datetime('today'), freq='D').tolist()  # DatetimeIndex를 리스트로 변환
    progress_bar = st.progress(0)
    ii = 0
    for idate in dates:
        ii = ii + 1
        progress_bar.progress(ii/len(dates))
        indate = pd.to_datetime(idate.strftime('%Y%m%d'))
        dates.append(indate)
        re = daily_returns(indate, all_data)  # all_data를 사용하여 수익 계산
        returns.append(re)
    
    df = pd.DataFrame()
    df['date'] = dates
    df['returns'] = returns
    df['cumsum'] = df.returns.cumsum()
    
    # Streamlit을 통해 데이터프레임 출력
    st.set_page_config(layout="wide")  # 페이지 레이아웃을 넓게 설정
    st.dataframe(df[::-1], use_container_width=True)  # 데이터프레임을 가득 차게 출력

    # print(df)

if __name__ == "__main__":
    start = time.time()
    all_data = update_binance()  # 반환된 데이터를 저장
    print(time.time() - start)
    main(all_data)  # all_data를 main에 전달 