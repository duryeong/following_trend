import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as tb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def make_idx(df, r1=7, ad=14, limad=12, mc_ratio=1.01, wmean=4):
    try:
        if df is None or df.empty:
            return None
            
        df[f'rsi{r1}'] = tb.rsi(df['close'], length=r1)
        df[f'rsi{r1*2}'] = tb.rsi(df['close'], length=r1*2)
        df[f'rsi{r1*3}'] = tb.rsi(df['close'], length=r1*3)
        df[f'adx_{ad}'] = tb.adx(df['high'], df['low'], df['close'], length=ad).iloc[:,0]
        df[f'mean{wmean}'] = df.close.rolling(window=wmean).mean()
        df[f'mc_ratio_{mc_ratio}'] = df.close / df[f'mean{wmean}']
        
        is_up = []
        for idf in df.iloc:
            is_up.append(idf[f'rsi{r1}'] > idf[f'rsi{r1*2}'] > idf[f'rsi{r1*3}'] and 
                        idf[f'adx_{ad}'] > limad and 
                        idf[f'mc_ratio_{mc_ratio}'] < mc_ratio and 
                        idf.close > idf[f'mean{wmean}'])
        df['is_up'] = is_up
        return df
    except Exception as e:
        st.error(f"지표 계산 중 오류: {str(e)}")
        return None

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
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y")  # 최근 1개월 데이터
                
                if df.empty:
                    continue
                    
                df.columns = [c.lower() for c in df.columns]
                
                # 파라미터 적용하여 지표 계산
                params = eval(row['best_param'])
                df = make_idx(df, **params)
                
                if df is None:
                    continue
                
                # 날짜 처리 수정
                if selected_date:
                    target_date = pd.Timestamp(selected_date).tz_localize('UTC')
                    # 주말인 경우 이전 금요일로 조정
                    while target_date.weekday() > 4:
                        target_date -= pd.Timedelta(days=1)
                    
                    # DataFrame의 인덱스를 UTC로 변환
                    df.index = df.index.tz_convert('UTC')
                    valid_dates = df[df.index <= target_date]
                    
                    if not valid_dates.empty:
                        last_date = valid_dates.index[-1]
                        if valid_dates.loc[last_date, 'is_up']:
                            recommended_stocks.append({
                                'ticker': ticker,
                                'close': df.loc[last_date, 'close'],
                                'date': last_date.date()
                            })
                else:
                    # 최신 데이터만 확인
                    target_date = df.index[-1]
                    if df.loc[target_date, 'is_up']:
                        recommended_stocks.append({
                            'ticker': ticker,
                            'close': df.loc[target_date, 'close'],
                            'date': target_date.date()
                        })
                
            except Exception as e:
                st.warning(f"{ticker} 처리 중 오류: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        return recommended_stocks
        
    except Exception as e:
        st.error(f"추천 종목 분석 중 오류: {str(e)}")
        return []

def main():
    st.title('주식 포트폴리오 분석기')
    
    # 날짜 선택 위젯
    selected_date = st.date_input(
        "분석 날짜 선택 (선택하지 않으면 최신 데이터 사용)",
        value=None
    )
    
    # 분석 시작 버튼 제거하고 자동 분석 실행
    st.text(selected_date)
    with st.spinner('분석 중...'):
        recommended_stocks = get_stock_recommendations(selected_date)
        
        if recommended_stocks:
            st.success(f'분석 완료! 추천 종목 수: {len(recommended_stocks)}개')
            
            # 결과를 데이터프레임으로 변환
            df_results = pd.DataFrame(recommended_stocks)
            df_results['close'] = df_results['close'].round(2)
            df_results.columns = ['종목코드', '종가', '기준일']
            df_results.set_index('종목코드', inplace=True)
            df_results = df_results[['종가', '기준일']]
            
            # 결과 표시 - index=False 추가
            st.dataframe(df_results, use_container_width=True)
            
            # CSV 다운로드 버튼
            # csv = df_results.to_csv(index=False)
            # st.download_button(
            #     label="CSV 다운로드",
            #     data=csv,
            #     file_name="recommended_stocks.csv",
            #     mime="text/csv"
            )
        else:
            st.warning('추천 종목이 없습니다.')

if __name__ == "__main__":
    main() 