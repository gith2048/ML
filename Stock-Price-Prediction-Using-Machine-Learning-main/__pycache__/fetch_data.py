import yfinance as yf
import pandas as pd

# Tickers: Yahoo Finance symbols mapped to user-friendly names
tickers = {
    'BTC-USD': 'BITCOIN',
    'ETH-USD': 'ETHEREUM',
    'HDFCBANK.NS': 'HDFC',
    'ICICIBANK.NS': 'ICICI',
    '^NSEBANK': 'BANKNIFTY',
    '^NSEI': 'NIFTY',
    'GC=F': 'GOLD',
    'GBPUSD=X': 'GBPUSD',
    'USDINR=X': 'USDINR'
    
    

}

start_date = '2012-01-01'
end_date = '2025-05-02'
__cached__="stock_data.csv"

df = pd.DataFrame()

for ticker, name in tickers.items():
    print(f"Downloading {name}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    df[name] = data['Close']

df.reset_index(inplace=True)
df.to_csv("stock_data.csv", index=False)
print("✅ Saved data to stock_data.csv")