# pages/9_period_cum_returns.py
import json
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf
from datetime import date

warnings.filterwarnings("ignore")
st.set_page_config(page_title="구간 누적 수익률 (월~월)", layout="wide")


# =========================
# 1) 지표/신호 계산
# =========================
def make_idx(df: pd.DataFrame, r1=7, ad=14, limad=12, mc_ratio=1.01, wmean=4) -> pd.DataFrame:
    """
    df columns: ['open','high','low','close'] (DatetimeIndex)
    'is_up' 신호:
      - RSI 단기>중기>장기
      - ADX > limad
      - close/MA(wmean) < mc_ratio
      - close > MA(wmean)
    """
    out = df.copy()

    out[f"rsi{r1}"] = ta.rsi(out["close"], length=int(r1))
    out[f"rsi{r1*2}"] = ta.rsi(out["close"], length=int(r1 * 2))
    out[f"rsi{r1*3}"] = ta.rsi(out["close"], length=int(r1 * 3))

    adx_tbl = ta.adx(out["high"], out["low"], out["close"], length=int(ad))
    adx_col = f"ADX_{int(ad)}"
    if adx_tbl is not None and adx_col in adx_tbl.columns:
        out[f"adx_{ad}"] = adx_tbl[adx_col]
    else:
        adx_candidates = [] if adx_tbl is None else [c for c in adx_tbl.columns if c.startswith("ADX_")]
        out[f"adx_{ad}"] = adx_tbl[adx_candidates[0]] if adx_candidates else 0.0

    out[f"mean{wmean}"] = out["close"].rolling(window=int(wmean)).mean()
    ratio_col = f"mc_ratio_{mc_ratio}"
    out[ratio_col] = out["close"] / out[f"mean{wmean}"]

    cond_rsi = (out[f"rsi{r1}"] > out[f"rsi{r1*2}"]) & (out[f"rsi{r1*2}"] > out[f"rsi{r1*3}"])
    cond_adx = out[f"adx_{ad}"] > float(limad)
    cond_mc  = out[ratio_col] < float(mc_ratio)
    cond_ma  = out["close"] > out[f"mean{wmean}"]

    out["is_up"] = (cond_rsi & cond_adx & cond_mc & cond_ma).fillna(False)
    return out


# =========================
# 2) CSV/가격/신호 로딩
# =========================
@st.cache_data(show_spinner=False)
def load_stock_info(csv_path: str, top_n: int) -> pd.DataFrame:
    """
    CSV에서 종목 정보를 로드합니다.
    - 'universe.csv'와 같은 티커 목록 CSV: 'tickers' 컬럼만 사용하고, 기본 파라미터를 적용합니다.
    - 기존 분석 CSV: 'best_value'로 정렬하여 상위 N개 종목을 선택하고, 'best_param'을 사용합니다.
    """
    df = pd.read_csv(csv_path)

    # 'tickers' 컬럼 처리
    if "tickers" not in df.columns:
        if len(df.columns) == 1:
            df.columns = ["tickers"]
        else:
            # 'ticker' 또는 'symbol'과 같은 대안 컬럼명 확인
            candidates = [c for c in df.columns if c.lower() in ['ticker', 'tickers', 'symbol']]
            if candidates:
                df = df.rename(columns={candidates[0]: "tickers"})
            else:
                raise ValueError("CSV 파일에 'tickers' 컬럼이 필요합니다.")

    # 분석 데이터('best_value', 'best_param') 존재 여부에 따른 처리
    if "best_value" in df.columns and "best_param" in df.columns:
        # 기존 로직: best_value로 정렬 후 상위 N개 선택
        return df.sort_values("best_value", ascending=False).head(top_n).reset_index(drop=True)
    else:
        # 유니버스 CSV 처리: best_param이 없으면 기본값 사용을 위해 빈 dict 할당
        if "best_param" not in df.columns:
            df["best_param"] = "{}"
        return df.head(top_n).reset_index(drop=True)


def parse_params(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return eval(str(s), {"__builtins__": {}})


@st.cache_data(show_spinner=False)
def download_prices(tickers: Tuple[str, ...], period="36mo", interval="1d") -> Dict[str, pd.DataFrame]:
    """
    yfinance로 여러 종목 일괄 다운로드 (OHLC)
    반환: {ticker: df_ohlc[['open','high','low','close']]}
    """
    if len(tickers) == 0:
        return {}
    data = yf.download(list(tickers), period=period, interval=interval,
                       group_by="ticker", auto_adjust=False, threads=True, progress=False)
    res = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.levels[0]:
                continue
            df_t = data[t].copy()
            df_t.columns = [c.lower() for c in df_t.columns]
            if {"open", "high", "low", "close"}.issubset(df_t.columns):
                res[t] = df_t[["open", "high", "low", "close"]].dropna()
    else:
        df_t = data.copy()
        df_t.columns = [c.lower() for c in df_t.columns]
        if {"open", "high", "low", "close"}.issubset(df_t.columns):
            res[tickers[0]] = df_t[["open", "high", "low", "close"]].dropna()
    return res


@st.cache_data(show_spinner=False)
def build_signal_table(stock_info: pd.DataFrame, period="36mo") -> Dict[str, pd.DataFrame]:
    """
    각 티커에 대해 OHLC + is_up 컬럼을 포함한 DF 반환
    """
    tickers = tuple(stock_info["tickers"].astype(str))
    price_map = download_prices(tickers, period=period, interval="1d")
    result = {}
    for _, row in stock_info.iterrows():
        t = str(row["tickers"]).strip()
        if t not in price_map:
            continue
        params = parse_params(row["best_param"])
        df = price_map[t].copy()
        sig = make_idx(
            df,
            r1=int(params.get("r1", 7)),
            ad=int(params.get("ad", 14)),
            limad=float(params.get("limad", 12)),
            mc_ratio=float(params.get("mc_ratio", 1.01)),
            wmean=int(params.get("wmean", 4)),
        )
        out = df.join(sig[["is_up"]], how="left")
        out["is_up"] = out["is_up"].fillna(False)
        result[t] = out
    return result


# =========================
# 3) 포트폴리오 수익률 (전일 신호 보유 규칙 + 현금 1몫)
# =========================
def compute_daily_portfolio_returns(data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    규칙:
      - 보유 여부: held_t = is_up.shift(1)
      - 매 거래일, held=True인 종목들의 일수익률 평균에 '현금 1몫' 포함한 동일가중 적용
      - r_t = (1/(K_t+1)) * sum_{i∈Held_t}(r_i), 현금 수익률은 0
      - K_t = 그날 보유(Held) 종목 수
    """
    if not data:
        return pd.Series(dtype=float)

    # 공용 날짜 인덱스
    all_dates = sorted(set().union(*[df.index for df in data.values()]))
    idx = pd.DatetimeIndex(all_dates)

    # 종목별 일수익률 & 보유마스크(전일 신호)
    rets, held = {}, {}
    for t, df in data.items():
        r = df["close"].pct_change().reindex(idx).fillna(0.0)
        rets[t] = r
        held_mask = df["is_up"].reindex(idx).fillna(False).shift(1).fillna(False)
        held[t] = held_mask

    ret_df = pd.DataFrame(rets, index=idx)
    held_df = pd.DataFrame(held, index=idx)

    K = held_df.sum(axis=1)
    weights = pd.Series(0.0, index=idx)
    nz = K > 0
    weights.loc[nz] = 1.0 / (K.loc[nz] + 1.0)

    active_ret_sum = (ret_df.where(held_df, 0.0)).sum(axis=1)
    port_daily = weights * active_ret_sum
    port_daily.name = "portfolio_daily_return"
    return port_daily


# =========================
# 4) 월~월 구간 슬라이싱 & 누적
# =========================
def period_slice(series: pd.Series, y1: int, m1: int, y2: int, m2: int) -> pd.Series:
    """
    시작 년월(y1,m1)은 그 달 1일, 끝 년월(y2,m2)은 그 달의 말일로 설정하여
    [start, end] 구간의 시리즈만 반환
    """
    if series.empty:
        return series
    s = series.copy()
    s.index = pd.to_datetime(s.index)

    start = pd.Timestamp(year=y1, month=m1, day=1)
    end = pd.Timestamp(year=y2, month=m2, day=1) + pd.offsets.MonthEnd(1)

    return s[(s.index >= start) & (s.index <= end)]


def cumulative_from_start(daily_ret: pd.Series) -> pd.Series:
    """
    (1 + r).cumprod() - 1
    """
    if daily_ret.empty:
        return daily_ret
    eq = (1.0 + daily_ret).cumprod()
    return eq - 1.0


# =========================
# 5) Streamlit UI — 시작/끝 '월' 선택 (버튼 없이 자동 반영)
# =========================
st.title("구간 누적 수익률 (월~월) — is_up 전략 + 현금 1몫 동일비중")

with st.sidebar:
    st.header("설정")
    csv_path = st.text_input("분석용 CSV 경로", value="pages/universe.csv")
    top_n = st.number_input("상위 N종목 (best_value 기준)", min_value=1, max_value=200, value=30, step=1)
    period = st.selectbox("가격 다운로드 기간", ["12mo", "18mo", "24mo", "36mo"], index=3)

    # 시작/끝 '월' 선택(연/월 selectbox)
    this_year = date.today().year
    this_month = date.today().month
    years = list(range(this_year - 10, this_year + 1))   # 최근 10년
    months = list(range(1, 13))

    st.subheader("조회 구간 (월~월)")
    col_sy, col_sm = st.columns(2)
    with col_sy:
        start_year = st.selectbox("시작 연도", years, index=len(years) - 1, key="start_year_sel")
    with col_sm:
        start_month = st.selectbox("시작 월", months, index=1-1 if this_month==1 else this_month-1, key="start_month_sel")

    col_ey, col_em = st.columns(2)
    with col_ey:
        end_year = st.selectbox("끝 연도", years, index=len(years) - 1, key="end_year_sel")
    with col_em:
        end_month = st.selectbox("끝 월", months, index=this_month-1, key="end_month_sel")

# ---- 위젯 변경 시 자동 재계산 ----
try:
    # 유효성: 시작이 끝보다 나중이면 스왑
    if (start_year, start_month) > (end_year, end_month):
        st.warning("시작 년월이 끝 년월보다 늦습니다. 값을 자동 보정했습니다.")
        start_year, start_month, end_year, end_month = end_year, end_month, start_year, start_month

    with st.spinner("티커/파라미터 로딩..."):
        stock_info = load_stock_info(csv_path, int(top_n))
        if stock_info.empty:
            st.warning("선정된 종목이 없습니다.")
            st.stop()

    with st.spinner("가격 다운로드 & 신호 계산..."):
        sig_map = build_signal_table(stock_info, period=period)
        if not sig_map:
            st.warning("가격/신호 데이터가 비어 있습니다.")
            st.stop()

    with st.spinner("포트폴리오 일수익률 계산..."):
        daily_port = compute_daily_portfolio_returns(sig_map)
        if daily_port.empty:
            st.warning("일 수익률이 계산되지 않았습니다.")
            st.stop()

    # 선택 구간(시작월 1일 ~ 끝월 말일)의 일별 수익률 & 누적
    daily_sel = period_slice(daily_port, start_year, start_month, end_year, end_month)
    cum_sel = cumulative_from_start(daily_sel)

    # 화면 표기용 날짜
    start_date_disp = pd.Timestamp(year=start_year, month=start_month, day=1).date()
    end_date_disp = (pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(1)).date()

    st.success(f"조회 구간: {start_date_disp} ~ {end_date_disp}")

    if daily_sel.empty:
        st.warning("선택한 구간에 거래일 데이터가 없습니다.")
        st.stop()

    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("일별 누적 수익률(%)")
        st.line_chart((cum_sel * 100.0).rename("누적수익률(%)"))
    with colB:
        period_total = float((1.0 + daily_sel).prod() - 1.0)
        st.metric("구간 누적 수익률", f"{period_total*100:.2f}%")
        mdd = (cum_sel.cummax() - cum_sel).max() if not cum_sel.empty else 0.0
        st.metric("구간 최대낙폭(MDD)", f"{float(mdd)*100:.2f}%")

    # 표 (일별 수익률, 누적수익률)
    table = pd.DataFrame({
        "일자": cum_sel.index.date,
        "일수익률(%)": (daily_sel * 100).round(3).values,
        "누적수익률(%)": (cum_sel * 100).round(3).values
    }).set_index("일자")
    st.subheader("일별 수익률 표")
    st.dataframe(table, use_container_width=True)

    # 다운로드
    csv_bytes = table.to_csv(index=True).encode("utf-8")
    st.download_button(
        "CSV 다운로드 (일별 수익률/누적수익률)",
        data=csv_bytes,
        file_name=f"daily_cum_returns_{start_year}_{start_month:02d}_to_{end_year}_{end_month:02d}.csv",
        mime="text/csv"
    )

    # 참고: 구간 내 보유(전일 신호) 종목 개요
    with st.expander("구간 내 보유(전일 신호) 종목 개요"):
        idx_range = pd.DatetimeIndex(pd.to_datetime(table.index))
        held_map = {}
        for t, df in sig_map.items():
            held = df["is_up"].reindex(idx_range).fillna(False).shift(1).fillna(False)
            held_map[t] = held
        held_df = pd.DataFrame(held_map, index=idx_range)
        st.write("일자별 보유 종목 수")
        st.dataframe(held_df.sum(axis=1).rename("보유종목수").to_frame(), use_container_width=True)
        last_day = idx_range[-1]
        held_list = [t for t in held_df.columns if bool(held_df.loc[last_day, t])]
        st.write(f"{last_day.date()} 보유 종목 수: {len(held_list)}")
        if len(held_list) > 0:
            st.write(", ".join(held_list))

except Exception as e:
    st.error(f"오류: {type(e).__name__}: {e}")