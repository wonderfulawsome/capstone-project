#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install yfinance


# In[3]:


# S&P 500 상위 100개 주식의 티커 리스트 (예시로 사용)
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'JPM', 'V', 
    'PG', 'T', 'MA', 'DIS', 'XOM', 'BAC', 'VZ', 'INTC', 'WMT', 'MRK', 
    'PFE', 'CSCO', 'HD', 'KO', 'CVX', 'PEP', 'CMCSA', 'ORCL', 'ABBV', 'PM',
    'LLY', 'ABT', 'UNH', 'ACN', 'NFLX', 'MCD', 'MDT', 'NKE', 'IBM', 'TMO',
    'BA', 'MMM', 'AMGN', 'TXN', 'HON', 'SBUX', 'NEE', 'BMY', 'LIN', 'LOW',
    'C', 'CHTR', 'GILD', 'DHR', 'FIS', 'AMD', 'QCOM', 'BLK', 'UNP', 'INTU',
    'UPS', 'COST', 'CVS', 'LMT', 'GS', 'TGT', 'ISRG', 'NOW', 'AMT', 'DE',
    'MS', 'CAT', 'BKNG', 'ADBE', 'SPGI', 'SCHW', 'GE', 'CI', 'SYK', 'MO',
    'MDLZ', 'AXP', 'USB', 'TJX', 'D', 'MU', 'ZTS', 'RTX', 'PNC', 'NSC',
    'ANTM', 'GM', 'CB', 'CME', 'CL', 'PLD', 'LRCX', 'SO', 'BDX', 'ADI'
]


# In[4]:


import yfinance as yf
import pandas as pd

# 날짜 범위 설정
start_date = '2020-01-01'
end_date = '2020-12-31'

# 데이터 프레임을 생성하고 각 주식의 데이터를 다운로드
data = pd.DataFrame()
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    data[ticker] = stock_data['Adj Close']

# 데이터 확인
print(data.head())

# 데이터를 CSV 파일로 저장
data.to_csv('stock_data.csv')


# ### per 다운로드후 파일로 저장

# In[5]:


financial_info = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        financial_info[ticker] = {
            'PER': info.get('trailingPE', None),  # None을 기본값으로 사용하여 값이 없는 경우 None을 반환
        }
    except ValueError:  # 정보를 가져오는 중 문제가 발생한 경우 예외 처리
        financial_info[ticker] = {
            'PER': None
        }
        print(f"Failed to fetch data for {ticker}")

print(financial_info)

# 딕셔너리를 DataFrame으로 변환
df_financial_info = pd.DataFrame(list(financial_info.items()), columns=['Ticker', 'PER'])

# DataFrame 확인
print(df_financial_info)

# DataFrame을 CSV 파일로 저장
df_financial_info.to_csv('per_values.csv', index=False)


# ### 베타값 다운로드후 저장
# 
# 직접 계산

# In[6]:


import yfinance as yf
import pandas as pd
import numpy as np

# 날짜 범위 설정
start_date = '2020-01-01'
end_date = '2020-12-31'

# S&P 500 지수 데이터 가져오기
market_index = yf.Ticker('^GSPC')
market_data = market_index.history(start=start_date, end=end_date)
market_data = market_data['Close'].resample('M').ffill().pct_change()

# 값을 저장할 딕셔너리 초기화
beta_values = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    try:
        # 주식 데이터 가져오기
        stock_data = stock.history(start=start_date, end=end_date)
        stock_data = stock_data['Close'].resample('M').ffill().pct_change()
        
        # 주식과 시장 지수의 월간 수익률 계산
        combined_data = pd.concat([stock_data, market_data], axis=1).dropna()
        combined_data.columns = ['Stock', 'Market']
        
        # 베타 계산
        cov_matrix = np.cov(combined_data['Stock'], combined_data['Market'])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        beta_values[ticker] = beta
    except Exception as e:  # 정보를 가져오는 중 문제가 발생한 경우 예외 처리
        beta_values[ticker] = None
        print(f"Failed to fetch data for {ticker}: {e}")

# 딕셔너리를 DataFrame으로 변환
df_beta_values = pd.DataFrame(list(beta_values.items()), columns=['Ticker', 'Beta'])

# DataFrame 확인
print(df_beta_values)


# yfinance 에서 베타값 가져오는 코드

# In[7]:


# 베타값을 저장할 딕셔너리 초기화
beta_values = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        beta_values[ticker] = info.get('beta', None)  # 베타값 수집
    except ValueError:  # 정보를 가져오는 중 문제가 발생한 경우 예외 처리
        beta_values[ticker] = None
        print(f"Failed to fetch data for {ticker}")

# 딕셔너리를 DataFrame으로 변환
df_beta_values = pd.DataFrame(list(beta_values.items()), columns=['Ticker', 'Beta'])

# DataFrame 확인
print(df_beta_values)

# DataFrame을 CSV 파일로 저장
df_beta_values.to_csv('df_beta_values.csv', index=False)


# In[8]:


df_beta_values


# In[9]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 결측치가 있는 행 제거
df_beta_values.dropna(subset=['Beta'], inplace=True)

# 클러스터링을 위해 'Beta' 열만 선택
X = df_beta_values[['Beta']]

# KMeans 모델 생성 및 학습
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

# 클러스터 레이블 추가
df_beta_values['Cluster'] = kmeans.labels_

# 결과 출력
print(df_beta_values)


# 베타값이 클러스터링 테스트 하여 분산이 잘 되는지 확인

# In[10]:


# 클러스터링 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(df_beta_values['Ticker'], df_beta_values['Beta'], c=df_beta_values['Cluster'], cmap='viridis')
plt.xlabel('Ticker')
plt.ylabel('Beta')
plt.title('K-means Clustering of Beta Values with k=5')
plt.colorbar(label='Cluster')
plt.show()


# 적절하게 클러스터가 형성되는것을 볼수있다

# ### 배당 수익률 다운로드

# In[11]:


# 배당수익률을 저장할 딕셔너리 초기화
dividend_yields = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        # 배당수익률 수집, 배당수익률이 없는 경우 None으로 처리
        dividend_yields[ticker] = info.get('dividendYield', None) * 100 if info.get('dividendYield') is not None else None
    except ValueError:  # 정보를 가져오는 중 문제가 발생한 경우 예외 처리
        dividend_yields[ticker] = None
        print(f"Failed to fetch data for {ticker}")

# 딕셔너리를 DataFrame으로 변환
df_dividend_yields = pd.DataFrame(list(dividend_yields.items()), columns=['Ticker', 'DividendYield'])

# DataFrame 확인
print(df_dividend_yields)


# In[12]:


# DataFrame을 CSV 파일로 저장
df_dividend_yields.to_csv('dividend_yields.csv', index=False)


# ### 거래량 다운로드

# In[13]:


# 각 티커에 대해 데이터 다운로드
data = yf.download(tickers, start="2023-01-01", end="2023-04-01")

# 거래량만 추출
volume_data = data['Volume']

volume_data.head()


# 거래량 평균 계산

# In[14]:


# 각 티커의 평균 거래량 계산
average_volumes = volume_data.mean()

print(average_volumes)


# In[15]:


# 평균 거래량 데이터를 'volume'이라는 열 이름으로 데이터 프레임 생성
average_volume_df = pd.DataFrame(average_volumes, columns=['volume'])

print(average_volume_df)


# 거래량 데이터에서 결측치 제거하기

# In[16]:


average_volumes.isnull().sum()


# In[17]:


# 결측치가 있는 행 제거
average_volumes_cleaned = average_volumes.dropna()

average_volumes_cleaned.isnull().sum()


# In[18]:


# 인덱스를 해제하고 열 이름 변경
average_volumes = average_volumes_cleaned.reset_index().rename(columns={'index': 'Ticker', 'volume': 'Volume'})

# 결과 출력
print(average_volumes)


# In[19]:


# DataFrame을 CSV 파일로 저장
average_volumes.to_csv('average_volumes.csv', index=False)


# ### 변동성 다운로드

# In[21]:


import yfinance as yf
import pandas as pd

# 데이터를 다운로드할 기간 설정
start_date = '2023-01-01'
end_date = '2023-12-31'

# 종목별로 데이터 다운로드 및 변동성 계산을 위한 빈 DataFrame 생성
volatility_df = pd.DataFrame()

# 각 종목에 대해 데이터 다운로드 및 변동성 계산
for ticker in tickers:
    # 데이터 다운로드
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # 일별 수익률 계산
    daily_returns = data['Adj Close'].pct_change()
    
    # 변동성 계산 (일별 수익률의 표준편차)
    volatility = daily_returns.std()
    
    # 결과를 DataFrame에 추가
    new_row = pd.DataFrame({'Ticker': [ticker], 'Volatility': [volatility]})
    volatility_df = pd.concat([volatility_df, new_row], ignore_index=True)

# DataFrame 출력
print(volatility_df)


# In[22]:


# DataFrame을 CSV 파일로 저장
volatility_df.to_csv('volatility_df.csv', index=False)


# ### RSI 지수 다운로드

# In[23]:


import yfinance as yf
import pandas as pd

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 데이터 다운로드 기간 설정
start_date = '2023-01-01'
end_date = '2023-12-31'

# 결과 저장을 위한 DataFrame 생성
rsi_summary_df = pd.DataFrame()

for ticker in tickers:
    # yfinance를 통해 주식 데이터 다운로드
    data = yf.download(ticker, start=start_date, end=end_date)

    # RSI 계산
    rsi = calculate_rsi(data['Adj Close'])

    # 최근 50일의 RSI 평균 계산
    rsi_mean = rsi.tail(50).mean()

    # 결과를 새로운 DataFrame에 추가
    new_row = pd.DataFrame({'Ticker': [ticker], 'RSI': [rsi_mean]})
    rsi_summary_df = pd.concat([rsi_summary_df, new_row], ignore_index=True)

# 결과 출력
print(rsi_summary_df)



# In[24]:


# DataFrame을 CSV 파일로 저장
rsi_summary_df.to_csv('average_rsi_df.csv', index=False)

