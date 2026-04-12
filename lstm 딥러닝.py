import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt

# 1. 시가총액 상위 종목 추출
def get_top_5_tickers():
    try:
        df_krx = fdr.StockListing('KRX')
        candidates = ['Marcap', 'MarCap', '시가총액']
        target_col = next((col for col in candidates if col in df_krx.columns), 'Marcap')
        top_5 = df_krx.sort_values(target_col, ascending=False).head(5)
        return top_5['Code'].tolist(), top_5['Name'].tolist()
    except:
        return ["005930", "000660", "373220", "207940", "005380"], ["삼성전자", "SK하이닉스", "LG엔솔", "삼성바이오", "현대차"]

# 2. 특정 기간 데이터 수집 및 지표 계산
def fetch_and_prepare(ticker, start_dt, end_dt):
    try:
        df = fdr.DataReader(ticker, start_dt, end_dt)
        if df.empty or len(df) < 40: return None
        
        # 기술적 지표: 볼린저 밴드
        ma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = ma20 + (std20 * 2)
        df['Lower_Band'] = ma20 - (std20 * 2)
        
        return df[['Close', 'Upper_Band', 'Lower_Band']].dropna()
    except:
        return None

# 3. LSTM 모델 구성
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# 4. 메인 실행 루프
def main():
    # 설정: 2020년 ~ 2021년
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    window_size = 10
    
    codes, names = get_top_5_tickers()
    
    for ticker, name in zip(codes, names):
        print(f"\n{'='*50}\n[백테스트 시작] {name} ({ticker}) | 기간: {start_date} ~ {end_date}")
        
        df = fetch_and_prepare(ticker, start_date, end_date)
        if df is None: continue

        # 데이터 전처리
        scaler = MinMaxScaler()
        features = ['Close', 'Upper_Band', 'Lower_Band']
        scaled_data = scaler.fit_transform(df[features])

        # 학습 데이터 생성 (X: 10일치 데이터, y: 다음날 상승 여부)
        X, y = [], []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i: i + window_size])
            y.append(1 if scaled_data[i + window_size, 0] > scaled_data[i + window_size - 1, 0] else 0)
        
        X, y = np.array(X), np.array(y)
        
        # 학습/테스트 분리 (7:3 비율)
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 모델 학습
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

        # 테스트 구간(나머지 30%)에 대해 매일매일 예측 및 매매 시뮬레이션
        test_df = df.iloc[window_size + split:].copy()
        test_preds = model.predict(X_test, verbose=0)
        
        test_df['Prob'] = test_preds
        # 전략: (종가가 상단밴드 위) AND (AI 상승 확률 > 0.5)
        test_df['Signal'] = np.where((test_df['Close'] > test_df['Upper_Band']) & (test_df['Prob'] > 0.5), 1, 0)
        
        # 수익률 계산
        test_df['Daily_Return'] = test_df['Close'].pct_change()
        test_df['Strategy_Return'] = test_df['Signal'].shift(1) * test_df['Daily_Return']
        
        # 누적 수익률 계산
        market_cum = (1 + test_df['Daily_Return'].fillna(0)).cumprod().iloc[-1] - 1
        strategy_cum = (1 + test_df['Strategy_Return'].fillna(0)).cumprod().iloc[-1] - 1
        
        print(f"▶ [결과] 단순 보유 수익률: {market_cum:.2%}")
        print(f"▶ [결과] AI 전략 수익률: {strategy_cum:.2%}")
        
        if strategy_cum > market_cum:
            print(f"✨ {name}: AI 전략이 시장보다 높은 수익을 냈습니다!")
        else:
            print(f"📉 {name}: 단순 보유가 더 유리했습니다.")

if __name__ == "__main__":
    main()