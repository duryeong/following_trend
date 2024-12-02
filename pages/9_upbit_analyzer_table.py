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

    df['pre_close'] = df.close.shift(1)
    df['differ'] = (df['close']-df['pre_close'])/df['pre_close']*100

    return df

def load_upbit_info():
    return pd.read_csv('coin_anal_upbit_for_optimization_241126.csv')

def get_upbit_recommendations(selected_date=None, upbit_dict=None):
    try:
        upbit_info = load_upbit_info()
        upbit_info = upbit_info.head(30)
        
        recommended_upbits = []
        
        for idx, (_, row) in enumerate(upbit_info.iterrows()):
            ticker = row['tickers']
            if row['best_param'] is None: continue
            try:
                if ticker in upbit_dict:
                    df = upbit_dict[ticker]
                else:
                    continue
                
                if df.empty:
                    continue
                
                # 파라미터 적용하여 지표 계산
                params = eval(row['best_param'])
                df = make_idx(df, r1=params['r1'], ad=params['ad'], limad=params['limad'], mc_ratio=params['mc_ratio'],wmean=params['wmean'])

                if df is None:
                    continue
                
                # 날짜 처리 수정
                if selected_date:
                    target_date = pd.to_datetime(selected_date) + pd.Timedelta(9, unit='h')
                    df.index = pd.to_datetime(df.index)
                    
                    # 선택된 날짜 이하의 데이터만 필터링
                    valid_data = df[df.index <= target_date]
                    
                    if not valid_data.empty:
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
                            
                            # 수익률 계산
                            start_price = valid_data.loc[period_start, 'close']
                            end_price = valid_data.loc[last_date, 'close']
                            returns = ((end_price - start_price) / start_price) * 100
                            
                            recommended_upbits.append({
                                'ticker': ticker,
                                'close': valid_data.loc[last_date, 'close'],
                                'date': last_date.date(),
                                'is_up':valid_data.loc[last_date, 'is_up'],
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
                                last_false_idx = false_indices[-2]  + 1
                        period_start = current_period.index[last_false_idx]
                            
                        start_price = df.loc[period_start, 'close']
                        end_price = df.loc[target_date, 'close']
                        returns = ((end_price - start_price) / start_price) * 100
                        
                        recommended_upbits.append({
                            'ticker': ticker,
                            'close': df.loc[target_date, 'close'],
                            'date': target_date.date(),
                            'is_up': df.loc[target_date, 'is_up'],
                            'returns': round(returns, 2)
                        })
                
            except Exception as e:
                print(f"종목 {ticker} 처리 중 상세 오류:\n{type(e).__name__}: {str(e)}")
                import traceback
                print(f"스택 트레이스:\n{traceback.format_exc()}")
                continue
        
        return recommended_upbits
        
    except Exception as e:
        print(f"상세 오류 정보:\n{type(e).__name__}: {str(e)}")
        import traceback
        print(f"스택 트레이스:\n{traceback.format_exc()}")
        return []

def daily_returns(idate, upbit_dict):
    recommended_upbits = get_upbit_recommendations(idate, upbit_dict)
    # 결과를 데이터프레임으로 변환
    if not recommended_upbits: return np.nan
    df_results = pd.DataFrame(recommended_upbits)
    df_results['close'] = df_results['close'].round(2)
    df_results.columns = ['종목코드', '종가', '기준일', 'is_up', '수익률(%)']
    df_results.set_index('종목코드', inplace=True)
    df_results = df_results[['종가', '기준일', 'is_up', '수익률(%)']]
    
    # 평균 수익률 계산
    if len(df_results[df_results.is_up == False]) > 0:
        average_returns = df_results[df_results.is_up == False]['수익률(%)'].sum()
    else:
        average_returns = np.nan
    return average_returns

def update_upbit():
    upbit_info = load_upbit_info()
    upbit_info = upbit_info.head(30)

    upbit_dict = {}  # 딕셔너리 초기화

    def fetch_data(ticker):
        print(f"처리 중: {ticker}")  # 처리 중인 티커 출력
        upbit = ccxt.upbit()  # upbit 인스턴스 생성

        # 최대 재시도 횟수 설정
        max_retries = 5
        for attempt in range(max_retries):
            try:
                df = upbit.fetch_ohlcv(ticker, timeframe='1d')  # OHLCV 데이터 가져오기
                df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # 데이터프레임으로 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 타임스탬프 변환
                df.set_index('timestamp', inplace=True)  # 타임스탬프를 인덱스로 설정
                df.columns = [ic.lower() for ic in list(df.columns)]  # 열 이름 소문자로 변환
                upbit_dict[ticker] = df  # 딕셔너리에 데이터 저장
                print(f"완료: {ticker}")  # 처리 완료된 티커 출력
                break  # 성공적으로 처리되면 루프 종료
            except ccxt.RateLimitExceeded as e:
                print(f"요청 제한 초과: {e}. {attempt + 1}/{max_retries} 시도 중...")
                time.sleep(2 ** attempt)  # 지수 백오프 대기
            except Exception as e:
                print(f"종목 {ticker} 처리 중 오류 발생: {str(e)}")
                break  # 다른 오류 발생 시 루프 종료

    # Streamlit 프로세스 바 초기화
    progress_bar = st.progress(0)  # 초기값 0으로 설정

    # 병렬 처리
    total_tickers = len(upbit_info['tickers'])  # 총 티커 수
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(fetch_data, ticker): ticker for ticker in upbit_info['tickers']}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()  # 결과를 기다림
            progress_bar.progress((i + 1) / total_tickers)  # 프로세스 바 업데이트

    return upbit_dict  # 딕셔너리 반환

def main(upbit_dict):  # upbit_dict 매개변수 추가
    dates = pd.date_range(pd.to_datetime('today') - pd.to_timedelta(30, unit='D'), pd.to_datetime('today'), freq='D').tolist()  # DatetimeIndex를 리스트로 변환
    returns = []
    progress_bar = st.progress(0)  # 초기값 0으로 설정
    ii = 0
    for idate in dates:
        indate = pd.to_datetime(idate.strftime('%Y%m%d'))
        returns.append(daily_returns(indate, upbit_dict))  # upbit_dict 전달
        ii += 1
        progress_bar.progress(ii / len(dates))

    df = pd.DataFrame()
    df['date'] = dates
    df['returns'] = returns
    df['sum_re'] = df.returns.cumsum()
    
    # Streamlit을 사용하여 데이터프레임 출력
    st.title("Upbit Analyzer Results")  # 제목 추가
    st.dataframe(df[::-1], use_container_width=True)  # 데이터프레임을 좌우로 가득 차게 출력

if __name__ == "__main__":
    start = time.time()
    upbit_dict = update_upbit()  # 딕셔너리에 데이터 저장
    print(time.time() - start)
    main(upbit_dict)  # 딕셔너리 전달 