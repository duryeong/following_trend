import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.stats import linregress
import matplotlib.dates as mdates
from datetime import timedelta
import streamlit as st
import time
plt.rc('font', family='NanumGothic')  # 예: 나눔고딕 폰트 설정 (시스템에 설치된 한글 폰트 사용)

# Streamlit 설정
st.set_page_config(page_title="BTC/USDT 실시간 캔들 차트", layout="wide")

# Binance Exchange 객체 생성 (미국 접근용)
exchange = ccxt.binanceus()  # 미국 사용자를 위한 binanceusdm API로 변경


# 데이터 가져오는 함수
def fetch_data():
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)  # 최신 200개의 1분봉 데이터 가져오기
    data = pd.DataFrame(ohlcv, columns=['Timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data.set_index('Timestamp', inplace=True)
    return data


# 실시간 차트 생성 함수
def plot_chart(data):
    fig, ax = plt.subplots()
    ax.set_title(f"BTC/USDT 1분봉 실시간 캔들 차트 - {pd.to_datetime('today')}")

    # 최근 60분 데이터에서 마지막 캔들을 제외하고 상단/하단 회귀 직선 계산
    recent_data = data[-60:]
    if len(recent_data) >= 60:
        data_to_fit = recent_data[:-1]  # 마지막 캔들을 제외한 59개 데이터
        last_candle = recent_data.iloc[-1]  # 마지막 캔들만 선택
        recent_close = data_to_fit['close']
        recent_high = data_to_fit['high']
        recent_low = data_to_fit['low']
        x_recent = mdates.date2num(recent_close.index)  # 59개 데이터의 datetime을 matplotlib 숫자 형식으로 변환

        # 기울기와 절편 계산 (59개 종가 기준)
        slope, intercept, _, _, _ = linregress(x_recent, recent_close.values)

        # 상단 및 하단 회귀 직선 계산
        shift_up = (recent_high - (slope * x_recent + intercept)).max()
        intercept_max = intercept + shift_up
        shift_down = (recent_low - (slope * x_recent + intercept)).min()
        intercept_min = intercept + shift_down

        # 전체 데이터에 대한 x축 설정
        x_full = mdates.date2num(data.index)

        # 상단 및 하단 회귀 직선, 외곽 회귀 직선 계산
        regression_line_max = slope * x_full + intercept_max
        regression_line_min = slope * x_full + intercept_min
        regression_line_max_outer = regression_line_max*1.001
        regression_line_min_outer = regression_line_min*0.999

        # 기본 회귀 직선 색상 설정 (회색)
        max_color = 'gray'
        min_color = 'gray'

        # 마지막 캔들의 종가가 회귀 직선 사이에 있는지 확인하여 색상 변경
        if regression_line_max_outer[-1] >= last_candle['close'] >= regression_line_max[-1]:
            max_color = 'gold'
        if regression_line_max_outer[-1] < last_candle['close']:
            max_color = 'red'
        if regression_line_min_outer[-1] <= last_candle['close'] <= regression_line_min[-1]:
            min_color = 'gold'
        if regression_line_min_outer[-1] > last_candle['close']:
            min_color = 'red'

        # mplfinance 캔들 차트 생성
        mpf.plot(
            data[-60:],
            type='candle',
            style='charles',
            ax=ax,
            volume=False,
            show_nontrading=True
        )

        # 회귀 직선과 외곽 직선을 차트에 추가
        ax.plot(data.index[-60:], regression_line_max[-60:], color=max_color, linestyle='--', label='상단 회귀 직선')
        ax.plot(data.index[-60:], regression_line_min[-60:], color=min_color, linestyle='--', label='하단 회귀 직선')
        ax.plot(data.index[-60:], regression_line_max_outer[-60:], color=max_color, linestyle=':', label='상단 외곽 회귀 직선 (0.1% 위)')
        ax.plot(data.index[-60:], regression_line_min_outer[-60:], color=min_color, linestyle=':', label='하단 외곽 회귀 직선 (0.1% 아래)')
        ax.legend()

        # x축 범위 설정
        buffer_start = data.index[-60] - timedelta(minutes=10)
        buffer_end = data.index[-1] + timedelta(minutes=10)
        ax.set_xlim([buffer_start, buffer_end])

    return fig


# Streamlit 실시간 업데이트 루프
st.title("BTC/USDT 실시간 캔들 차트")
st.write("Binance의 1분봉 데이터를 기반으로 회귀 직선을 표시한 실시간 차트입니다.")
chart_area = st.empty()

while True:
    data = fetch_data()
    fig = plot_chart(data)
    chart_area.pyplot(fig)
    time.sleep(1)  # 5초마다 업데이트
