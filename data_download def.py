# data_download.py의 주요 로직을 함수로 정의
def download_data(tickers):
    # 주가 데이터를 다운로드하고 파일로 저장하는 로직 (예시)
    import yfinance as yf

    data = yf.download(tickers, start='2023-01-01', end='2023-12-31')
    data.to_csv('downloaded_data.csv')  # 예시로 데이터를 파일에 저장
    return data
