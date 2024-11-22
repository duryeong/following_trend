import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as tb
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

@st.cache_data
def load_stock_info():
    return pd.read_csv('yfinance_anal_v2.csv')

def get_stock_recommendations(selected_date=None):
    try:
        # 캐시된 데이터 사용
        stock_info = load_stock_info()
        stock_info = stock_info.sort_values(by='best_value', ascending=False).head(30)
        
        recommended_stocks = []
        total_stocks = len(stock_info)
        
        # 진행 상황 표시 추가
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (_, row) in enumerate(stock_info.iterrows()):
            ticker = row['tickers']
            try:
                status_text.text(f'분석 중: {ticker} ({idx+1}/{total_stocks})')
                progress_bar.progress((idx + 1) / total_stocks)
                
                # 주식 데이터 가져오기 (타임아웃 설정)
                # stock = yf.Ticker(ticker)
                # df = stock.history(period="1y")  # 최근 1년 데이터
                stock_data = yf.Ticker(ticker)
                df = stock_data.history(interval='1d', period='6mo')
                df.columns = [ic.lower() for ic in list(df.columns)]
                df = df[['open', 'high', 'low', 'close']]
                
                if df.empty:
                    continue
                    
                df.columns = [c.lower() for c in df.columns]
                
                # 파라미터 적용하여 지표 계산
                params = eval(row['best_param'])
                df = make_idx(df, r1=params['r1'], ad=params['ad'], limad=params['limad'], mc_ratio=params['mc_ratio'], wmean=params['wmean'])

                
                if df is None:
                    continue
                
                # 날짜 처리 수정
                if selected_date:
                    target_date = pd.Timestamp(selected_date).tz_localize('UTC') + pd.Timedelta(9, unit='h')
                    df.index = df.index.tz_convert('UTC')
                    
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
                            
                            recommended_stocks.append({
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
                        
                        recommended_stocks.append({
                            'ticker': ticker,
                            'close': df.loc[target_date, 'close'],
                            'date': target_date.date(),
                            'is_up': df.loc[target_date, 'is_up'],
                            'returns': round(returns, 2)
                        })
                
            except Exception as e:
                st.warning(f"{ticker} 처리 중 오류: {str(e)}")
                print(f"종목 {ticker} 처리 중 상세 오류:\n{type(e).__name__}: {str(e)}")
                import traceback
                print(f"스택 트레이스:\n{traceback.format_exc()}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        return recommended_stocks
        
    except Exception as e:
        st.error(f"추천 종목 분석 중 오류: {str(e)}")
        print(f"상세 오류 정보:\n{type(e).__name__}: {str(e)}")
        import traceback
        print(f"스택 트레이스:\n{traceback.format_exc()}")
        return []

def main():
    st.title('주식 포트폴리오 분석기')
    
    # session_state 초기화 수정
    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = datetime.now().date()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_date = st.date_input(
            "분석 날짜 선택 (선택하지 않으면 최신 데이터 사용)",
            value=st.session_state.selected_date
        )
        # date_input의 값이 변경될 때 session_state 업데이트
        st.session_state.selected_date = selected_date
    
    with col2:
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("◀ 이전날"):
                if st.session_state.selected_date:
                    new_date = st.session_state.selected_date - timedelta(days=1)
                    while new_date.weekday() > 4:
                        new_date = new_date - timedelta(days=1)
                    st.session_state.selected_date = new_date
                    st.rerun()
        with col2_2:
            if st.button("금일"):
                today = datetime.now().date()
                while today.weekday() > 4:  # 주말인 경우 직전 금요일로 설정
                    today -= timedelta(days=1)
                st.session_state.selected_date = today
                st.rerun()
    
    with col3:
        if st.button("다음날 ▶"):
            if st.session_state.selected_date:
                new_date = st.session_state.selected_date + timedelta(days=1)
                while new_date.weekday() > 4:
                    new_date = new_date + timedelta(days=1)
                st.session_state.selected_date = new_date
                st.rerun()
    
    # session_state에서 날짜 가져오기
    if 'selected_date' in st.session_state:
        selected_date = st.session_state.selected_date
    
    # 분석 시작 버튼 제거하고 자동 분석 실행
    st.text(selected_date)
    with st.spinner('분석 중...'):
        recommended_stocks = get_stock_recommendations(selected_date)
        
        if recommended_stocks:
            st.success(f'분석 완료! 추천 종목 수: {len(recommended_stocks)}개')
            
            # 결과를 데이터프레임으로 변환
            df_results = pd.DataFrame(recommended_stocks)
            df_results['close'] = df_results['close'].round(2)
            df_results.columns = ['종목코드', '종가', '기준일', 'is_up', '수익률(%)']
            df_results.set_index('종목코드', inplace=True)
            df_results = df_results[['종가', '기준일', 'is_up', '수익률(%)']]
            
            # 결과 표시 - index=False 추가
            st.dataframe(df_results, use_container_width=True)
            
            # CSV 다운로드 버튼
            # csv = df_results.to_csv(index=False)
            # st.download_button(
            #     label="CSV 다운로드",
            #     data=csv,
            #     file_name="recommended_stocks.csv",
            #     mime="text/csv"
            # )
        else:
            st.warning('추천 종목이 없습니다.')

    # 디버깅을 위한 세션 상태 출력
    st.write("현재 선택된 날짜:", st.session_state.selected_date)

if __name__ == "__main__":
    main() 