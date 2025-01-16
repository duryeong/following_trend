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

top = 2
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

    return pd.concat([eth_df, btc_df, n_df])
def get_upbit_recommendations(selected_date=None, upbit_dict=None):
    try:
        upbit_info = load_upbit_info()
        upbit_info = upbit_info.head(top)
        
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
                df = make_idx(df, r1=params['r1'], ad=params['ad'], limad=params['limad'], mc_ratio=params['mc_ratio'], wmean=params['wmean'])

                if df is None:
                    continue
                
                # 날짜 처리 수정
                if selected_date:
                    target_date = pd.to_datetime(selected_date) + pd.Timedelta(9, unit='h')
                    df.index = pd.to_datetime(df.index)
                    
                    # 선택된 날짜 이하의 데이터만 필터링
                    valid_data = df[df.index <= target_date]
                    
                    returns = np.nan
                    if len(valid_data) < 2:
                        continue
                    if not valid_data.is_up.values[-1] and valid_data.is_up.values[-2]:
                        # 역으로 탐색하여 is_up이 True에서 False로 바뀌는 부분 찾기
                        for i in range(len(valid_data) - 1, 0, -1):
                            if valid_data.is_up.values[i] == True and valid_data.is_up.values[i - 1] == False:
                                # 해당 인덱스에서의 처리
                                # print(f"변화 발생: {valid_data.index[i]}에서 is_up이 False로 변경됨")
                                # print(valid_data.is_up.values[i])
                                # print(valid_data.is_up.values[i - 1])
                                returns = (valid_data.close.values[-1]/valid_data.close.values[i]-1)*100
                                break

                    recommended_upbits.append({
                        'ticker': ticker,
                        'close': valid_data.close.values[-1],
                        'date': target_date.date(),
                        'is_up': valid_data.is_up.values[-1],
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
    if not recommended_upbits: return np.nan, ''
    df_results = pd.DataFrame(recommended_upbits)
    df_results['close'] = df_results['close'].round(2)
    df_results.columns = ['종목코드', '종가', '기준일', 'is_up', '수익률(%)']
    df_results.set_index('종목코드', inplace=True)
    df_results = df_results[['종가', '기준일', 'is_up', '수익률(%)']]
    
    # 보유 종목 처리 수정
    keep = ''
    if df_results.is_up.any():  # is_up이 True인 경우가 있는지 확인
        keep = "/".join(df_results.index[df_results.is_up].tolist())

    # 평균 수익률 계산
    if len(df_results[df_results.is_up == False]) > 0:
        average_returns = df_results[df_results.is_up == False]['수익률(%)'].sum()
    else:
        average_returns = np.nan
    return average_returns, keep

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
    for ikey in upbit_dict.keys():
        upbit_dict[ikey] = make_idx(upbit_dict[ikey])

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
    df['sum_returns'] = df.returns.cumprod()
    df['keeps'] = np.array(keeps)
    # df.to_csv('table_upbit.csv')
    return df

def web_main(df):
    if st.button('rerun'):
        st.rerun()
    st.title(f'Following Trend: ETH, BTC')
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