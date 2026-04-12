import streamlit as st
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import plotly.graph_objects as go
from datetime import datetime

# --- 1. 데이터 및 지표 계산 (정밀 보정) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    return 100 - (100 / (1 + (ema_up / ema_down)))

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    df = fdr.DataReader(ticker, start_date, end_date)
    if df.empty: return None
    
    # 지표 계산
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Upper_Band'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['RSI'] = calculate_rsi(df['Close'])
    df['Vol_Avg5'] = df['Volume'].rolling(5).mean()
    
    # 변동성 계산 (단순 수익률 기준 표준편차로 변경하여 일관성 유지)
    df['Ret'] = df['Close'].pct_change()
    df['Volatility'] = df['Ret'].rolling(window=20).std()
    df['Vol_Hist_Avg'] = df['Volatility'].rolling(window=60).mean()
    
    return df.dropna()

# --- 2. LSTM 모델 ---
def train_lstm_model(df, features):
    if len(df) <= 20: return None, None
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    window_size = 10
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i: i + window_size])
        y.append(1 if scaled_data[i + window_size, 0] > scaled_data[i + window_size - 1, 0] else 0)
    if not X: return None, None
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(32, return_sequences=True),
        Dropout(0.1),
        LSTM(16),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model, scaler

# --- 3. 실행 UI ---
st.set_page_config(page_title="백테스터", layout="wide")
st.title("백테스팅 결과")

st.sidebar.header("⚙️ 설정")
ticker = st.sidebar.text_input("종목코드", value="005930")
start_date = st.sidebar.date_input("학습 시작일", datetime(2021, 1, 1))
split_date = st.sidebar.date_input("테스트 시작일", datetime(2025, 1, 1))
end_date = st.sidebar.date_input("종료일", datetime.today())
prob_threshold = st.sidebar.slider("lstm모델 예측 확률 기준", 0.45, 0.65, 0.50)
vol_multiplier = st.sidebar.slider("변동성 제한(배수)", 1.0, 3.0, 1.5)

if st.sidebar.button("결과 확인"):
    features = ['Close', 'Upper_Band', 'MA20', 'RSI', 'Volume', 'Volatility']
    full_df = get_stock_data(ticker, start_date, end_date)
    
    if full_df is not None:
        train_df = full_df[full_df.index < pd.to_datetime(split_date)]
        test_df = full_df[full_df.index >= pd.to_datetime(split_date)]
        
        model, scaler = train_lstm_model(train_df, features)
        
        if model:
            # 테스트 구간 데이터 준비
            scaled_test = scaler.transform(test_df[features])
            X_test = [scaled_test[i:i+10] for i in range(len(scaled_test)-10)]
            preds = model.predict(np.array(X_test), verbose=0)
            
            # 결과 프레임 (시작일 정렬)
            res_df = test_df.iloc[10:].copy()
            res_df['Prob'] = preds
            res_df['Vol_Guard'] = res_df['Volatility'] < (res_df['Vol_Hist_Avg'] * vol_multiplier)
            
            res_df['Signal'] = np.where(
                (res_df['Prob'] > prob_threshold) & (res_df['Close'] > res_df['MA20']) & 
                (res_df['RSI'] < 80) & (res_df['Vol_Guard']), 1, 0
            )
            
            # 수익률 계산 핵심 보정 (매수 시점 t+1 반영)
            res_df['Market_Ret'] = res_df['Close'].pct_change().fillna(0)
            res_df['Strategy_Ret'] = (res_df['Signal'].shift(1).fillna(0) * res_df['Market_Ret'])
            
            # 누적 수익률 (시작점을 1.0으로 강제 고정)
            res_df['Cum_Market'] = (1 + res_df['Market_Ret']).cumprod()
            res_df['Cum_Strategy'] = (1 + res_df['Strategy_Ret']).cumprod()
            
            # 수익률 정규화 (그래프 시작점을 0%로)
            market_final = (res_df['Cum_Market'].iloc[-1] - 1) * 100
            strategy_final = (res_df['Cum_Strategy'].iloc[-1] - 1) * 100

            col1, col2 = st.columns(2)
            col1.metric("시장 수익률", f"{market_final:.2f}%")
            col2.metric("전략 수익률", f"{strategy_final:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_df.index, y=(res_df['Cum_Market']-1)*100, name="시장(삼성전자 등)"))
            fig.add_trace(go.Scatter(x=res_df.index, y=(res_df['Cum_Strategy']-1)*100, name="AI 전략"))
            fig.update_layout(title="누적 수익률 (%) 비교", yaxis_title="수익률 (%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)