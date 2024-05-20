#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 각 파일을 로드
average_rsi_df = pd.read_csv('C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/average_rsi_df.csv')
average_volumes = pd.read_csv('C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/average_volumes.csv')
df_beta_values = pd.read_csv('C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/df_beta_values.csv')
dividend_yields = pd.read_csv('C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/dividend_yields.csv')
per_values = pd.read_csv('C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/per_values.csv')
volatility_df = pd.read_csv('C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/volatility_df.csv')

# 모든 데이터를 'Ticker' 기준으로 병합
data_frames = [average_rsi_df, average_volumes, df_beta_values, dividend_yields, per_values, volatility_df]
merged_df = pd.concat(data_frames, axis=1)
merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]  # 중복 열 제거

# 병합된 데이터 프레임 출력
merged_df.head()


# ### 배당수익률이 NaN인 행은 0으로 대체

# In[2]:


# 'DividendYield' 열의 결측치를 0으로 대체
merged_df['DividendYield'] = merged_df['DividendYield'].fillna(0)


# In[3]:


merged_df.rename(columns={'0': 'volume'}, inplace=True)


# ### 6개열에서 결측치가 한개라도 있는 행은 삭제

# In[4]:


# 결측치가 있는 행 삭제
cleaned_df = merged_df.dropna(subset=['PER', 'DividendYield', 'Beta','RSI','volume','Volatility'])

# 결과 확인
print(cleaned_df)


# 13개 행이 삭제됨

# ### 범주형 변수 인코딩

# In[5]:


# 데이터 유형 확인
print("Data types before conversion:")
print(cleaned_df[['PER', 'DividendYield', 'Beta','RSI','volume','Volatility']].dtypes)


# PER의 { } 안의 수만 추출 하도록 가공

# In[6]:


import ast
import pandas as pd

def extract_per_value(per_string):
    if pd.isna(per_string):
        return float('nan')  # 입력 값이 NaN인 경우, NaN 반환

    try:
        # 문자열을 사전으로 변환
        per_dict = ast.literal_eval(per_string)
        # 'PER' 키의 값을 float으로 변환하여 반환
        return float(per_dict['PER']) if per_dict['PER'] is not None else float('nan')
    except (ValueError, SyntaxError, KeyError, TypeError):
        # 변환에 실패한 경우 NaN 반환
        return float('nan')

# 'PER' 열을 숫자형으로 변환
cleaned_df['PER'] = cleaned_df['PER'].apply(extract_per_value)

# 결과 확인
print(cleaned_df[['Ticker', 'PER']])


# In[7]:


cleaned_df


# ### 결측치 처리하기

# In[8]:


# 'PER' 열에서 NaN 값을 가진 행을 식별
nan_per_tickers = cleaned_df[cleaned_df['PER'].isna()]['Ticker']

# 결과 출력
print("NaN 이 있는 티커:")
print(nan_per_tickers)


# In[9]:


# 'PER' 열에서 NaN 값이 있는 행을 삭제
cleaned_df = cleaned_df.dropna(subset=['PER'])


# In[10]:


# 'PER' 열에서 NaN 값을 가진 행을 식별
nan_per_tickers = cleaned_df[cleaned_df['PER'].isna()]['Ticker']

# 결과 출력
print("NaN 이 있는 티커:")
print(nan_per_tickers)


# In[11]:


# 저장 경로 설정
save_path = 'C:/Users/82106/Desktop/경영 캡스톤 프로젝트2/cleaned_data.csv'

# 데이터 프레임을 CSV 파일로 저장
cleaned_df.to_csv(save_path, index=False)


# In[ ]:




