import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MovingAverageCrossover:
    """
    A trading strategy based on moving average crossovers.
    
    Buy signal: When short-term MA crosses above long-term MA
    Sell signal: When short-term MA crosses below long-term MA
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50, 
                 max_price: float = 200, min_volume: float = 1000000, 
                 volatility_window: int = 20):
        """
        Initialize the moving average crossover strategy with additional filters.
        
        Args:
            short_window (int): Period for short-term moving average (default: 20)
            long_window (int): Period for long-term moving average (default: 50)
            max_price (float): Maximum stock price filter (default: 200)
            min_volume (float): Minimum average daily volume filter (default: 1M)
            volatility_window (int): Window for volatility calculation (default: 20)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.max_price = max_price
        self.min_volume = min_volume
        self.volatility_window = volatility_window
        self.positions = []
        self.trades = []
        
    def get_market_stocks(self, max_price: float = None) -> List[str]:
        """
        Get a comprehensive list of US stocks under the specified price.
        
        Args:
            max_price (float): Maximum stock price to filter (default: self.max_price)
            
        Returns:
            List[str]: List of US stock symbols under the price threshold
        """
        if max_price is None:
            max_price = self.max_price
            
        print(f"Discovering US stocks under ${max_price}...")
        
        # Comprehensive list of US stocks from major exchanges (NYSE, NASDAQ, AMEX)
        all_us_stocks = [
            # US Large Cap (under $200)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'JPM', 'JNJ', 'PG', 'V', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM', 'NKE',
            'WMT', 'KO', 'PEP', 'ABT', 'TMO', 'AVGO', 'COST', 'ACN', 'DHR',
            'UNH', 'MA', 'LLY', 'PFE', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK',
            'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'RTX', 'LMT', 'NOC', 'GD',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'KMI', 'PSX', 'VLO',
            
            # US Mid Cap
            'AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX',
            'ADI', 'MRVL', 'SWKS', 'QRVO', 'MCHP', 'ON', 'STM', 'TSM',
            'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'MPC', 'OXY', 'DVN',
            'PXD', 'HES', 'APA', 'LIN', 'APD', 'FCX', 'NEM', 'NUE', 'STLD',
            'X', 'AA', 'ALB', 'LVS', 'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK',
            
            # US Small Cap
            'PLTR', 'SNOW', 'CRWD', 'NET', 'ZS', 'OKTA', 'TEAM', 'ZM',
            'SQ', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT', 'DASH',
            'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA', 'ATVI',
            'ZM', 'TEAM', 'OKTA', 'NET', 'CRWD', 'SNOW', 'PLTR', 'ZS',
            
            # US Healthcare Stocks
            'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'LLY', 'DHR', 'BMY',
            'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ALXN', 'ILMN', 'DXCM',
            'CVS', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            
            # US Financial Stocks
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
            'COF', 'USB', 'PNC', 'TFC', 'KEY', 'HBAN', 'RF', 'ZION',
            'MTB', 'FITB', 'CMA', 'CFG', 'WAL', 'PACW', 'SIVB', 'FRC',
            
            # US Consumer Stocks
            'AMZN', 'WMT', 'HD', 'COST', 'TGT', 'LOW', 'TJX', 'ROST',
            'NKE', 'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ', 'PEP', 'KO',
            'DIS', 'NFLX', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT',
            'DASH', 'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA',
            'ATVI', 'TSLA', 'F', 'GM', 'TM', 'HMC', 'NSANY', 'RACE',
            
            # US Industrial Stocks
            'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'RTX', 'LMT',
            'NOC', 'GD', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV',
            'UPS', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE',
            'UNP', 'CSX', 'NSC', 'KSU', 'CP', 'CNI', 'CNR', 'CP',
            
            # US Energy Stocks
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'KMI',
            'PSX', 'VLO', 'MPC', 'OXY', 'DVN', 'PXD', 'HES', 'APA',
            'EOG', 'PXD', 'COP', 'EOG', 'PXD', 'COP', 'EOG', 'PXD',
            
            # US Materials Stocks
            'LIN', 'APD', 'FCX', 'NEM', 'NUE', 'STLD', 'X', 'AA',
            'ALB', 'LVS', 'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK',
            'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF', 'LYB', 'MOS',
            
            # US Real Estate Stocks
            'AMT', 'CCI', 'DLR', 'EQIX', 'PLD', 'PSA', 'SPG', 'VICI',
            'WELL', 'VTR', 'HCP', 'HR', 'OHI', 'SBRA', 'DOC', 'MPW',
            
            # US Utilities Stocks
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'DTE', 'ED', 'EIX',
            'ETR', 'FE', 'NEE', 'DUK', 'SO', 'D', 'AEP', 'DTE',
            
            # US Communication Services
            'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'CHTR', 'VZ', 'T',
            'TMUS', 'S', 'LUMN', 'CTL', 'IRDM', 'SATS', 'VSAT', 'GILT',
            
            # US ETFs (for diversification)
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO',
            'XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB',
            'XLRE', 'XLU', 'XLC', 'EFA', 'EEM', 'AGG', 'TLT', 'GLD',
            'VEA', 'VWO', 'VEA', 'VWO', 'VEA', 'VWO', 'VEA', 'VWO',
            
            # Additional US Tech Stocks
            'ZM', 'TEAM', 'OKTA', 'NET', 'CRWD', 'SNOW', 'PLTR', 'ZS',
            'SQ', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT', 'DASH',
            'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA', 'ATVI',
            'MSTR', 'PANW', 'FTNT', 'ZS', 'CRWD', 'NET', 'OKTA', 'TEAM',
            'ZM', 'DOCU', 'TWLO', 'ESTC', 'MDB', 'DDOG', 'SNOW', 'PLTR',
            
            # US Biotech Stocks
            'GILD', 'REGN', 'VRTX', 'BIIB', 'ALXN', 'ILMN', 'DXCM',
            'AMGN', 'BMY', 'ABBV', 'MRK', 'PFE', 'JNJ', 'LLY', 'TMO',
            'ABT', 'DHR', 'GE', 'HON', 'MMM', 'RTX', 'LMT', 'NOC',
            
            # US Semiconductor Stocks
            'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC',
            'LRCX', 'ADI', 'MRVL', 'SWKS', 'QRVO', 'MCHP', 'ON', 'STM',
            'TSM', 'ASML', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MRVL', 'SWKS',
            
            # US Software Stocks
            'MSFT', 'ADBE', 'CRM', 'ORCL', 'SAP', 'INTU', 'ADP', 'CTSH',
            'WDAY', 'NOW', 'SPLK', 'VMW', 'RHT', 'ANSS', 'CDNS', 'SNPS',
            'PLTR', 'SNOW', 'CRWD', 'NET', 'ZS', 'OKTA', 'TEAM', 'ZM',
            
            # US Internet Stocks
            'GOOGL', 'META', 'AMZN', 'NFLX', 'TSLA', 'BABA', 'JD', 'PDD',
            'BIDU', 'NTES', 'TCEHY', 'NIO', 'XPENG', 'LI', 'XPEV', 'BILI',
            'DIDI', 'TME', 'VIPS', 'ZTO', 'YUMC', 'BABA', 'JD', 'PDD',
            
            # US Retail Stocks
            'AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'TJX', 'ROST',
            'NKE', 'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ', 'PEP', 'KO',
            'DIS', 'NFLX', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT',
            'DASH', 'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA',
            
            # US Auto Stocks
            'TSLA', 'F', 'GM', 'TM', 'HMC', 'NSANY', 'RACE', 'FCAU',
            'STLA', 'VWAGY', 'BMWYY', 'MBGYY', 'DDAIF', 'RACE', 'FCAU',
            
            # US Aerospace & Defense
            'BA', 'RTX', 'LMT', 'NOC', 'GD', 'LHX', 'TDG', 'AJRD',
            'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS',
            
            # US Banking Stocks
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
            'AXP', 'COF', 'USB', 'PNC', 'TFC', 'KEY', 'HBAN', 'RF',
            'ZION', 'MTB', 'FITB', 'CMA', 'CFG', 'WAL', 'PACW', 'SIVB',
            
            # US Insurance Stocks
            'UNH', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            'CVS', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            
            # US Transportation Stocks
            'UPS', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE',
            'UNP', 'CSX', 'NSC', 'KSU', 'CP', 'CNI', 'CNR', 'CP',
            
            # US Energy Services
            'SLB', 'HAL', 'BKR', 'KMI', 'PSX', 'VLO', 'MPC', 'OXY',
            'DVN', 'PXD', 'HES', 'APA', 'EOG', 'PXD', 'COP', 'EOG',
            
            # US Mining Stocks
            'FCX', 'NEM', 'NUE', 'STLD', 'X', 'AA', 'ALB', 'LVS',
            'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK', 'LIN', 'APD',
            
            # US Chemical Stocks
            'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF',
            'LYB', 'MOS', 'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN',
        ]
        
        # Remove duplicates while preserving order
        unique_stocks = list(dict.fromkeys(all_us_stocks))
        
        print(f"  Checking prices for {len(unique_stocks)} US stocks...")
        
        # Filter stocks by current price
        affordable_stocks = []
        checked_count = 0
        
        for symbol in unique_stocks:
            checked_count += 1
            if checked_count % 50 == 0:
                print(f"    Progress: {checked_count}/{len(unique_stocks)} stocks checked")
                
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    price = info['regularMarketPrice']
                    if price <= max_price:
                        affordable_stocks.append(symbol)
                        print(f"    ✓ {symbol}: ${price:.2f}")
                        
            except Exception as e:
                # Skip stocks that can't be checked
                continue
                
        print(f"  Found {len(affordable_stocks)} US stocks under ${max_price}")
        
        if len(affordable_stocks) == 0:
            print("  Warning: No stocks found under the price threshold. Using fallback list...")
            # Fallback to a smaller list of known affordable stocks
            fallback_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                'JPM', 'JNJ', 'PG', 'V', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM', 'NKE',
                'WMT', 'KO', 'PEP', 'ABT', 'TMO', 'AVGO', 'COST', 'ACN', 'DHR',
                'UNH', 'MA', 'LLY', 'PFE', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK',
                'AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX',
                'PLTR', 'SNOW', 'CRWD', 'NET', 'ZS', 'OKTA', 'TEAM', 'ZM',
                'SQ', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT', 'DASH',
                'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA', 'ATVI',
            ]
            return fallback_stocks
        
        return affordable_stocks
    
    def get_stocks_by_market_cap(self, max_price: float = None, min_market_cap: float = 1e9) -> List[str]:
        """
        Get US stocks filtered by market capitalization and price.
        
        Args:
            max_price (float): Maximum stock price (default: self.max_price)
            min_market_cap (float): Minimum market cap in USD (default: 1B)
            
        Returns:
            List[str]: List of US stock symbols meeting criteria
        """
        if max_price is None:
            max_price = self.max_price
            
        print(f"Getting US large-cap stocks with market cap > ${min_market_cap/1e9:.1f}B and price < ${max_price}...")
        
        # Focus on US large-cap stocks
        us_large_cap_stocks = [
            # US Large Cap
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'JPM', 'JNJ', 'PG', 'V', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM', 'NKE',
            'WMT', 'KO', 'PEP', 'ABT', 'TMO', 'AVGO', 'COST', 'ACN', 'DHR',
            'UNH', 'MA', 'LLY', 'PFE', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK',
            'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'RTX', 'LMT', 'NOC', 'GD',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'KMI', 'PSX', 'VLO',
            'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'MPC', 'OXY', 'DVN',
            'PXD', 'HES', 'APA', 'LIN', 'APD', 'FCX', 'NEM', 'NUE', 'STLD',
            'X', 'AA', 'ALB', 'LVS', 'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK',
            'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF', 'LYB', 'MOS',
            'AMT', 'CCI', 'DLR', 'EQIX', 'PLD', 'PSA', 'SPG', 'VICI',
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'DTE', 'ED', 'EIX',
            'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'CHTR', 'VZ', 'T',
            'TMUS', 'S', 'LUMN', 'CTL', 'IRDM', 'SATS', 'VSAT', 'GILT',
            'GILD', 'REGN', 'VRTX', 'BIIB', 'ALXN', 'ILMN', 'DXCM',
            'AMGN', 'BMY', 'ABBV', 'MRK', 'PFE', 'JNJ', 'LLY', 'TMO',
            'ABT', 'DHR', 'GE', 'HON', 'MMM', 'RTX', 'LMT', 'NOC',
            'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC',
            'LRCX', 'ADI', 'MRVL', 'SWKS', 'QRVO', 'MCHP', 'ON', 'STM',
            'TSM', 'ASML', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MRVL', 'SWKS',
            'MSFT', 'ADBE', 'CRM', 'ORCL', 'SAP', 'INTU', 'ADP', 'CTSH',
            'WDAY', 'NOW', 'SPLK', 'VMW', 'RHT', 'ANSS', 'CDNS', 'SNPS',
            'PLTR', 'SNOW', 'CRWD', 'NET', 'ZS', 'OKTA', 'TEAM', 'ZM',
            'GOOGL', 'META', 'AMZN', 'NFLX', 'TSLA', 'BABA', 'JD', 'PDD',
            'BIDU', 'NTES', 'TCEHY', 'NIO', 'XPENG', 'LI', 'XPEV', 'BILI',
            'DIDI', 'TME', 'VIPS', 'ZTO', 'YUMC', 'BABA', 'JD', 'PDD',
            'AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'TJX', 'ROST',
            'NKE', 'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ', 'PEP', 'KO',
            'DIS', 'NFLX', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT',
            'DASH', 'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA',
            'TSLA', 'F', 'GM', 'TM', 'HMC', 'NSANY', 'RACE', 'FCAU',
            'STLA', 'VWAGY', 'BMWYY', 'MBGYY', 'DDAIF', 'RACE', 'FCAU',
            'BA', 'RTX', 'LMT', 'NOC', 'GD', 'LHX', 'TDG', 'AJRD',
            'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS',
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
            'AXP', 'COF', 'USB', 'PNC', 'TFC', 'KEY', 'HBAN', 'RF',
            'ZION', 'MTB', 'FITB', 'CMA', 'CFG', 'WAL', 'PACW', 'SIVB',
            'UNH', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            'CVS', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            'UPS', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE',
            'UNP', 'CSX', 'NSC', 'KSU', 'CP', 'CNI', 'CNR', 'CP',
            'SLB', 'HAL', 'BKR', 'KMI', 'PSX', 'VLO', 'MPC', 'OXY',
            'DVN', 'PXD', 'HES', 'APA', 'EOG', 'PXD', 'COP', 'EOG',
            'FCX', 'NEM', 'NUE', 'STLD', 'X', 'AA', 'ALB', 'LVS',
            'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK', 'LIN', 'APD',
            'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF',
            'LYB', 'MOS', 'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN',
        ]
        
        # Remove duplicates
        unique_stocks = list(dict.fromkeys(us_large_cap_stocks))
        
        print(f"  Checking prices for {len(unique_stocks)} US large-cap stocks...")
        
        # Filter stocks by current price
        affordable_stocks = []
        checked_count = 0
        
        for symbol in unique_stocks:
            checked_count += 1
            if checked_count % 30 == 0:
                print(f"    Progress: {checked_count}/{len(unique_stocks)} stocks checked")
                
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    price = info['regularMarketPrice']
                    if price <= max_price:
                        affordable_stocks.append(symbol)
                        print(f"    ✓ {symbol}: ${price:.2f}")
                        
            except Exception as e:
                # Skip stocks that can't be checked
                continue
                
        print(f"  Found {len(affordable_stocks)} US large-cap stocks under ${max_price}")
        
        if len(affordable_stocks) == 0:
            print("  Warning: No large-cap stocks found under the price threshold. Using fallback list...")
            # Fallback to a smaller list of known affordable large-cap stocks
            fallback_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                'JPM', 'JNJ', 'PG', 'V', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM', 'NKE',
                'WMT', 'KO', 'PEP', 'ABT', 'TMO', 'AVGO', 'COST', 'ACN', 'DHR',
                'UNH', 'MA', 'LLY', 'PFE', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK',
                'AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX',
            ]
            return fallback_stocks
        
        return affordable_stocks
    
    def get_stocks_by_sector(self, max_price: float = None) -> Dict[str, List[str]]:
        """
        Get US stocks organized by sector, filtered by price.
        
        Args:
            max_price (float): Maximum stock price (default: self.max_price)
            
        Returns:
            Dict[str, List[str]]: Dictionary of US sectors and their stocks
        """
        if max_price is None:
            max_price = self.max_price
            
        print(f"Organizing US stocks by sector under ${max_price}...")
        
        us_sector_stocks = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                'PYPL', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'MU',
                'AMAT', 'KLAC', 'LRCX', 'ADI', 'MRVL', 'SWKS', 'QRVO', 'MCHP',
                'ON', 'STM', 'TSM', 'PLTR', 'SNOW', 'CRWD', 'NET', 'ZS',
                'OKTA', 'TEAM', 'ZM', 'SQ', 'ROKU', 'SPOT', 'PINS', 'SNAP',
                'UBER', 'LYFT', 'DASH', 'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD',
                'TTWO', 'EA', 'ATVI', 'MSTR', 'PANW', 'FTNT', 'ZS', 'CRWD',
                'NET', 'OKTA', 'TEAM', 'ZM', 'DOCU', 'TWLO', 'ESTC', 'MDB',
                'DDOG', 'SNOW', 'PLTR', 'ORCL', 'SAP', 'INTU', 'ADP', 'CTSH',
                'WDAY', 'NOW', 'SPLK', 'VMW', 'RHT', 'ANSS', 'CDNS', 'SNPS',
                'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TCEHY', 'NIO', 'XPENG',
                'LI', 'XPEV', 'BILI', 'DIDI', 'TME', 'VIPS', 'ZTO', 'YUMC',
            ],
            'Healthcare': [
                'JNJ', 'PFE', 'ABT', 'TMO', 'LLY', 'DHR', 'UNH', 'BMY',
                'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ALXN', 'ILMN', 'DXCM',
                'CVS', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
                'ABBV', 'MRK', 'PFE', 'JNJ', 'LLY', 'TMO', 'ABT', 'DHR',
                'GE', 'HON', 'MMM', 'RTX', 'LMT', 'NOC', 'GILD', 'REGN',
                'VRTX', 'BIIB', 'ALXN', 'ILMN', 'DXCM', 'AMGN', 'BMY',
            ],
            'Financial': [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
                'COF', 'USB', 'PNC', 'TFC', 'KEY', 'HBAN', 'RF', 'ZION',
                'MTB', 'FITB', 'CMA', 'CFG', 'WAL', 'PACW', 'SIVB', 'FRC',
                'ZION', 'MTB', 'FITB', 'CMA', 'CFG', 'WAL', 'PACW', 'SIVB',
                'UNH', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
                'CVS', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'KMI',
                'PSX', 'VLO', 'MPC', 'OXY', 'DVN', 'PXD', 'HES', 'APA',
                'EOG', 'PXD', 'COP', 'EOG', 'PXD', 'COP', 'EOG', 'PXD',
                'SLB', 'HAL', 'BKR', 'KMI', 'PSX', 'VLO', 'MPC', 'OXY',
                'DVN', 'PXD', 'HES', 'APA', 'EOG', 'PXD', 'COP', 'EOG',
            ],
            'Consumer': [
                'AMZN', 'WMT', 'HD', 'COST', 'TGT', 'LOW', 'TJX', 'ROST',
                'NKE', 'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ', 'PEP', 'KO',
                'DIS', 'NFLX', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT',
                'DASH', 'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA',
                'ATVI', 'TSLA', 'F', 'GM', 'TM', 'HMC', 'NSANY', 'RACE',
                'STLA', 'VWAGY', 'BMWYY', 'MBGYY', 'DDAIF', 'RACE', 'FCAU',
            ],
            'Industrial': [
                'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'RTX', 'LMT',
                'NOC', 'GD', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV',
                'UPS', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE',
                'UNP', 'CSX', 'NSC', 'KSU', 'CP', 'CNI', 'CNR', 'CP',
                'BA', 'RTX', 'LMT', 'NOC', 'GD', 'LHX', 'TDG', 'AJRD',
                'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS',
            ],
            'Materials': [
                'LIN', 'APD', 'FCX', 'NEM', 'NUE', 'STLD', 'X', 'AA',
                'ALB', 'LVS', 'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK',
                'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF', 'LYB', 'MOS',
                'FCX', 'NEM', 'NUE', 'STLD', 'X', 'AA', 'ALB', 'LVS',
                'WY', 'IP', 'PKG', 'BLL', 'SEE', 'WRK', 'LIN', 'APD',
                'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF',
                'LYB', 'MOS', 'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN',
            ],
            'Real Estate': [
                'AMT', 'CCI', 'DLR', 'EQIX', 'PLD', 'PSA', 'SPG', 'VICI',
                'WELL', 'VTR', 'HCP', 'HR', 'OHI', 'SBRA', 'DOC', 'MPW',
            ],
            'Utilities': [
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'DTE', 'ED', 'EIX',
                'ETR', 'FE', 'NEE', 'DUK', 'SO', 'D', 'AEP', 'DTE',
            ],
            'Communication Services': [
                'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'CHTR', 'VZ', 'T',
                'TMUS', 'S', 'LUMN', 'CTL', 'IRDM', 'SATS', 'VSAT', 'GILT',
            ],
            'Transportation': [
                'UPS', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE',
                'UNP', 'CSX', 'NSC', 'KSU', 'CP', 'CNI', 'CNR', 'CP',
            ],
            'Banking': [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
                'AXP', 'COF', 'USB', 'PNC', 'TFC', 'KEY', 'HBAN', 'RF',
                'ZION', 'MTB', 'FITB', 'CMA', 'CFG', 'WAL', 'PACW', 'SIVB',
            ],
            'Insurance': [
                'UNH', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
                'CVS', 'ANTM', 'CI', 'HUM', 'CNC', 'DVA', 'WBA', 'MCK',
            ],
            'Aerospace & Defense': [
                'BA', 'RTX', 'LMT', 'NOC', 'GD', 'LHX', 'TDG', 'AJRD',
                'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS', 'AJRD', 'KTOS',
            ],
            'Automotive': [
                'TSLA', 'F', 'GM', 'TM', 'HMC', 'NSANY', 'RACE', 'FCAU',
                'STLA', 'VWAGY', 'BMWYY', 'MBGYY', 'DDAIF', 'RACE', 'FCAU',
            ],
            'Semiconductors': [
                'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC',
                'LRCX', 'ADI', 'MRVL', 'SWKS', 'QRVO', 'MCHP', 'ON', 'STM',
                'TSM', 'ASML', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MRVL', 'SWKS',
            ],
            'Software': [
                'MSFT', 'ADBE', 'CRM', 'ORCL', 'SAP', 'INTU', 'ADP', 'CTSH',
                'WDAY', 'NOW', 'SPLK', 'VMW', 'RHT', 'ANSS', 'CDNS', 'SNPS',
                'PLTR', 'SNOW', 'CRWD', 'NET', 'ZS', 'OKTA', 'TEAM', 'ZM',
            ],
            'Internet': [
                'GOOGL', 'META', 'AMZN', 'NFLX', 'TSLA', 'BABA', 'JD', 'PDD',
                'BIDU', 'NTES', 'TCEHY', 'NIO', 'XPENG', 'LI', 'XPEV', 'BILI',
                'DIDI', 'TME', 'VIPS', 'ZTO', 'YUMC', 'BABA', 'JD', 'PDD',
            ],
            'Retail': [
                'AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'TJX', 'ROST',
                'NKE', 'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ', 'PEP', 'KO',
                'DIS', 'NFLX', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'UBER', 'LYFT',
                'DASH', 'ABNB', 'HOOD', 'COIN', 'RBLX', 'TTD', 'TTWO', 'EA',
            ],
            'Biotech': [
                'GILD', 'REGN', 'VRTX', 'BIIB', 'ALXN', 'ILMN', 'DXCM',
                'AMGN', 'BMY', 'ABBV', 'MRK', 'PFE', 'JNJ', 'LLY', 'TMO',
                'ABT', 'DHR', 'GE', 'HON', 'MMM', 'RTX', 'LMT', 'NOC',
            ],
            'Chemical': [
                'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN', 'FMC', 'IFF',
                'LYB', 'MOS', 'LIN', 'APD', 'DD', 'DOW', 'DUP', 'EMN',
            ],
        }
        
        # Filter stocks by price for each sector
        filtered_sector_stocks = {}
        total_original = sum(len(stocks) for stocks in us_sector_stocks.values())
        total_filtered = 0
        
        print(f"  Checking prices for {total_original} US stocks across {len(us_sector_stocks)} sectors...")
        
        for sector, stocks in us_sector_stocks.items():
            affordable_stocks = []
            
            for symbol in stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        price = info['regularMarketPrice']
                        if price <= max_price:
                            affordable_stocks.append(symbol)
                            
                except Exception as e:
                    # Skip stocks that can't be checked
                    continue
            
            filtered_sector_stocks[sector] = affordable_stocks
            total_filtered += len(affordable_stocks)
            print(f"    {sector}: {len(affordable_stocks)} stocks under ${max_price}")
            
        print(f"  Found {total_filtered} total US stocks under ${max_price} across {len(us_sector_stocks)} sectors")
        
        return filtered_sector_stocks
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages, volatility, and volume indicators.
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close' prices and 'Volume'
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        df = data.copy()
        
        # Calculate moving averages
        df['SMA_Short'] = df['Close'].rolling(window=self.short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=self.long_window).mean()
        
        # Calculate volatility (rolling standard deviation of returns)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=self.volatility_window).std() * np.sqrt(252) * 100
        df['Volatility'] = df['Volatility'].fillna(0)  # Fill NaN values with 0
        
        # Calculate average daily volume
        df['Avg_Volume'] = df['Volume'].rolling(window=self.volatility_window).mean()
        df['Avg_Volume'] = df['Avg_Volume'].fillna(method='bfill').fillna(method='ffill')  # Fill NaN values
        
        # Calculate price filters
        df['Price_Filter'] = df['Close'] <= self.max_price
        df['Volume_Filter'] = df['Avg_Volume'] >= self.min_volume
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on moving average crossovers and filters.
        
        Args:
            data (pd.DataFrame): DataFrame with indicators
            
        Returns:
            pd.DataFrame: DataFrame with signals added
        """
        df = data.copy()
        
        # Initialize signal column
        df['Signal'] = 0
        
        # Generate buy signals (short MA crosses above long MA AND filters pass)
        buy_condition = (df['SMA_Short'] > df['SMA_Long']) & \
                       (df['Price_Filter'].astype(bool)) & \
                       (df['Volume_Filter'].astype(bool))
        df.loc[buy_condition, 'Signal'] = 1
        
        # Generate sell signals (short MA crosses below long MA)
        sell_condition = (df['SMA_Short'] < df['SMA_Long'])
        df.loc[sell_condition, 'Signal'] = -1
        
        # Create position changes (when signal changes)
        df['Position_Change'] = df['Signal'].diff()
        
        return df
    
    def backtest(self, symbol: str, start_date: str, end_date: str, 
                 initial_capital: float = 10000) -> Dict:
        """
        Backtest the moving average crossover strategy.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            initial_capital (float): Initial capital to invest
            
        Returns:
            Dict: Backtest results including performance metrics
        """
        # Download data
        print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Remove NaN values
        data = data.dropna()
        
        # Initialize portfolio tracking
        portfolio = pd.DataFrame(index=data.index)
        portfolio['Holdings'] = 0
        portfolio['Cash'] = initial_capital
        portfolio['Total'] = initial_capital
        
        position = 0
        entry_price = 0
        
        # Simulate trading
        for i in range(len(data)):
            current_price = float(data['Close'].iloc[i])
            signal = data['Position_Change'].iloc[i]
            current_date = data.index[i]
            
            if signal == 1 and position == 0:  # Buy signal
                position = 1
                entry_price = current_price
                shares = float(portfolio['Cash'].iloc[i-1]) / current_price
                portfolio.loc[current_date, 'Holdings'] = shares * current_price
                portfolio.loc[current_date, 'Cash'] = 0
                
                # Record trade
                self.trades.append({
                    'Date': current_date,
                    'Action': 'BUY',
                    'Price': current_price,
                    'Shares': shares
                })
                
            elif signal == -1 and position == 1:  # Sell signal
                position = 0
                shares = float(portfolio['Holdings'].iloc[i-1]) / current_price
                portfolio.loc[current_date, 'Holdings'] = 0
                portfolio.loc[current_date, 'Cash'] = shares * current_price
                
                # Record trade
                self.trades.append({
                    'Date': current_date,
                    'Action': 'SELL',
                    'Price': current_price,
                    'Shares': shares
                })
                
            else:  # Hold position
                if position == 1:
                    shares = float(portfolio['Holdings'].iloc[i-1]) / float(data['Close'].iloc[i-1])
                    portfolio.loc[current_date, 'Holdings'] = shares * current_price
                    portfolio.loc[current_date, 'Cash'] = 0
                else:
                    portfolio.loc[current_date, 'Holdings'] = 0
                    portfolio.loc[current_date, 'Cash'] = float(portfolio['Cash'].iloc[i-1])
            
            portfolio.loc[current_date, 'Total'] = (
                portfolio.loc[current_date, 'Holdings'] + 
                portfolio.loc[current_date, 'Cash']
            )
        
        # Calculate performance metrics
        final_value = float(portfolio['Total'].iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Buy and hold comparison
        buy_hold_return = (float(data['Close'].iloc[-1]) - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0]) * 100
        
        # Calculate additional metrics
        portfolio['Returns'] = portfolio['Total'].pct_change()
        sharpe_ratio = np.sqrt(252) * portfolio['Returns'].mean() / portfolio['Returns'].std()
        max_drawdown = (portfolio['Total'] / portfolio['Total'].cummax() - 1).min() * 100
        
        results = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'data': data,
            'portfolio': portfolio,
            'trades': self.trades
        }
        
        return results
    
    def screen_stocks(self, symbols: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Screen stocks based on price, volume, and volatility criteria.
        
        Args:
            symbols (List[str]): List of stock symbols to screen
            start_date (str): Start date for analysis (default: 6 months ago)
            end_date (str): End date for analysis (default: today)
            
        Returns:
            pd.DataFrame: DataFrame with screening results and recommendations
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        results = []
        
        for symbol in symbols:
            try:
                print(f"Screening {symbol}...")
                
                # Download data
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    print(f"No data found for {symbol}")
                    continue
                
                # Check if we have the required columns
                required_columns = ['Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    print(f"Missing required columns for {symbol}. Available: {list(data.columns)}")
                    continue
                
                # Calculate indicators
                data = self.calculate_indicators(data)
                data = data.dropna()
                
                if len(data) < 30:  # Need sufficient data
                    continue
                
                # Get latest values (ensure scalar values)
                latest = data.iloc[-1]
                avg_values = data.mean()
                
                # Extract scalar values from latest row
                latest_close = float(latest['Close'])
                latest_avg_volume = float(latest['Avg_Volume'])
                latest_volatility = float(latest['Volatility'])
                latest_sma_short = float(latest['SMA_Short'])
                latest_sma_long = float(latest['SMA_Long'])
                
                # Calculate screening criteria
                price_ok = bool(latest_close <= self.max_price)
                volume_ok = bool(latest_avg_volume >= self.min_volume)
                volatility_ok = bool(latest_volatility <= 50)  # Less than 50% annualized volatility
                
                # MA crossover status
                ma_bullish = bool(latest_sma_short > latest_sma_long)
                ma_trend = float((latest_sma_short - latest_sma_long) / latest_sma_long * 100)
                
                # Generate recommendation
                score = 0
                recommendation = "HOLD"
                reasons = []
                
                if price_ok:
                    score += 1
                    reasons.append("Price under $200")
                else:
                    reasons.append("Price too high")
                
                if volume_ok:
                    score += 1
                    reasons.append("Good volume")
                else:
                    reasons.append("Low volume")
                
                if volatility_ok:
                    score += 1
                    reasons.append("Reasonable volatility")
                else:
                    reasons.append("High volatility")
                
                if ma_bullish:
                    score += 2
                    reasons.append("Bullish MA crossover")
                else:
                    reasons.append("Bearish MA crossover")
                
                # Determine recommendation based on score
                if score >= 4:
                    recommendation = "STRONG BUY"
                elif score == 3:
                    recommendation = "BUY"
                elif score == 2:
                    recommendation = "WEAK BUY"
                elif score == 1:
                    recommendation = "HOLD"
                else:
                    recommendation = "SELL"
                
                results.append({
                    'Symbol': symbol,
                    'Current_Price': latest_close,
                    'Avg_Volume': latest_avg_volume,
                    'Volatility': latest_volatility,
                    'MA_Trend': ma_trend,
                    'Price_OK': price_ok,
                    'Volume_OK': volume_ok,
                    'Volatility_OK': volatility_ok,
                    'MA_Bullish': ma_bullish,
                    'Score': score,
                    'Recommendation': recommendation,
                    'Reasons': '; '.join(reasons)
                })
                
            except Exception as e:
                print(f"Error screening {symbol}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        return pd.DataFrame(results)
    
    def get_market_opportunities(self, discovery_method: str = 'comprehensive', 
                                max_stocks: int = 500) -> pd.DataFrame:
        """
        Discover market opportunities across US stocks under $200.
        
        Args:
            discovery_method (str): Method to discover stocks ('comprehensive', 'market_cap', 'sector')
            max_stocks (int): Maximum number of stocks to analyze
            
        Returns:
            pd.DataFrame: DataFrame with US market opportunities and recommendations
        """
        print("="*80)
        print("US MARKET OPPORTUNITY DISCOVERY")
        print("="*80)
        print(f"Discovery Method: {discovery_method}")
        print(f"Price Filter: < ${self.max_price}")
        print(f"Volume Filter: > {self.min_volume:,}")
        print(f"Volatility Filter: < 50%")
        print("="*80)
        
        # Discover stocks based on method
        if discovery_method == 'comprehensive':
            symbols = self.get_market_stocks()
        elif discovery_method == 'market_cap':
            symbols = self.get_stocks_by_market_cap()
        elif discovery_method == 'sector':
            sector_data = self.get_stocks_by_sector()
            symbols = []
            for sector, stocks in sector_data.items():
                symbols.extend(stocks)
                print(f"  {sector}: {len(stocks)} stocks")
        else:
            raise ValueError(f"Unknown discovery method: {discovery_method}")
        
        # Limit number of stocks to analyze
        if len(symbols) > max_stocks:
            print(f"Limiting analysis to {max_stocks} stocks (found {len(symbols)})")
            symbols = symbols[:max_stocks]
        
        print(f"\nAnalyzing {len(symbols)} US stocks for market opportunities...")
        print("="*80)
        
        # Screen the discovered stocks
        results = self.screen_stocks(symbols)
        
        if results.empty:
            print("No US stocks passed the screening criteria.")
            return results
        
        # Sort by score (highest first)
        results = results.sort_values('Score', ascending=False)
        
        # Print market opportunities
        print("\nTOP US MARKET OPPORTUNITIES:")
        print("-"*80)
        
        for _, row in results.head(20).iterrows():
            print(f"{row['Symbol']:8s} | ${float(row['Current_Price']):6.2f} | "
                  f"Vol: {float(row['Avg_Volume']):8.0f} | "
                  f"Volatility: {float(row['Volatility']):5.1f}% | "
                  f"MA Trend: {float(row['MA_Trend']):6.1f}% | "
                  f"Score: {int(row['Score']):1d} | "
                  f"{row['Recommendation']:12s}")
        
        print(f"\nFound {len(results)} total US opportunities out of {len(symbols)} analyzed stocks")
        
        # Show breakdown by recommendation
        print("\nBREAKDOWN BY RECOMMENDATION:")
        print("-"*40)
        recommendation_counts = results['Recommendation'].value_counts()
        for rec, count in recommendation_counts.items():
            print(f"{rec:12s}: {count:3d} stocks")
        
        return results
    
    def get_stock_recommendations(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get stock recommendations based on all criteria.
        
        Args:
            symbols (List[str]): List of symbols to analyze (default: use market discovery)
            
        Returns:
            pd.DataFrame: DataFrame with recommendations
        """
        if symbols is None:
            # Use market discovery instead of predefined list
            return self.get_market_opportunities()
        
        print("="*80)
        print("STOCK SCREENING AND RECOMMENDATIONS")
        print("="*80)
        print(f"Screening {len(symbols)} stocks...")
        print(f"Criteria: Price < ${self.max_price}, Volume > {self.min_volume:,}, Volatility < 50%")
        print("="*80)
        
        results = self.screen_stocks(symbols)
        
        if results.empty:
            print("No stocks passed the screening criteria.")
            return results
        
        # Sort by score (highest first)
        results = results.sort_values('Score', ascending=False)
        
        # Print recommendations
        print("\nTOP RECOMMENDATIONS:")
        print("-"*80)
        
        for _, row in results.head(10).iterrows():
            print(f"{row['Symbol']:6s} | ${float(row['Current_Price']):6.2f} | "
                  f"Vol: {float(row['Avg_Volume']):8.0f} | "
                  f"Volatility: {float(row['Volatility']):5.1f}% | "
                  f"MA Trend: {float(row['MA_Trend']):6.1f}% | "
                  f"Score: {int(row['Score']):1d} | "
                  f"{row['Recommendation']:12s}")
        
        print("\nDETAILED ANALYSIS:")
        print("-"*80)
        
        for _, row in results.iterrows():
            print(f"\n{row['Symbol']} - {row['Recommendation']}")
            print(f"  Price: ${float(row['Current_Price']):.2f}")
            print(f"  Avg Volume: {float(row['Avg_Volume']):,.0f}")
            print(f"  Volatility: {float(row['Volatility']):.1f}%")
            print(f"  MA Trend: {float(row['MA_Trend']):.1f}%")
            print(f"  Score: {int(row['Score'])}/5")
            print(f"  Reasons: {row['Reasons']}")
        
        return results
    
    def plot_results(self, results: Dict, save_plot: bool = False):
        """
        Plot the backtest results with additional indicators.
        
        Args:
            results (Dict): Backtest results from the backtest method
            save_plot (bool): Whether to save the plot to file
        """
        data = results['data']
        portfolio = results['portfolio']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 15))
        
        # Plot 1: Price and Moving Averages
        ax1.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
        ax1.plot(data.index, data['SMA_Short'], label=f'SMA {self.short_window}', alpha=0.8)
        ax1.plot(data.index, data['SMA_Long'], label=f'SMA {self.long_window}', alpha=0.8)
        
        # Highlight buy/sell signals
        buy_signals = data[data['Position_Change'] == 1]
        sell_signals = data[data['Position_Change'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color='green', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color='red', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'Price and Moving Averages - {results["symbol"]}')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume
        ax2.plot(data.index, data['Volume'], label='Daily Volume', alpha=0.7, color='purple')
        ax2.plot(data.index, data['Avg_Volume'], label=f'Avg Volume ({self.volatility_window}d)', alpha=0.8, color='orange')
        ax2.axhline(y=self.min_volume, color='red', linestyle='--', label='Min Volume Threshold', alpha=0.7)
        ax2.set_title('Trading Volume')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Portfolio Value
        ax3.plot(portfolio.index, portfolio['Total'], label='Portfolio Value', linewidth=2)
        ax3.axhline(y=results['initial_capital'], color='red', linestyle='--', 
                   label='Initial Capital', alpha=0.7)
        ax3.set_title('Portfolio Performance')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volatility
        ax4.plot(data.index, data['Volatility'], label='Volatility', alpha=0.7, color='brown')
        ax4.axhline(y=50, color='red', linestyle='--', label='Max Volatility (50%)', alpha=0.7)
        ax4.set_title('Annualized Volatility')
        ax4.set_ylabel('Volatility (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Returns
        ax5.plot(portfolio.index, portfolio['Returns'], label='Daily Returns', alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_title('Daily Portfolio Returns')
        ax5.set_ylabel('Returns')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Price Filter Status
        price_filter = data['Price_Filter'].astype(int)
        ax6.fill_between(data.index, 0, price_filter, alpha=0.3, color='green', label='Price OK')
        ax6.fill_between(data.index, price_filter, 1, alpha=0.3, color='red', label='Price Too High')
        ax6.set_title('Price Filter Status (< $200)')
        ax6.set_ylabel('Filter Status')
        ax6.set_ylim(0, 1)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'ma_crossover_{results["symbol"]}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """
        Print a summary of the backtest results.
        
        Args:
            results (Dict): Backtest results from the backtest method
        """
        print("\n" + "="*60)
        print(f"MOVING AVERAGE CROSSOVER STRATEGY RESULTS")
        print("="*60)
        print(f"Symbol: {results['symbol']}")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Short MA: {self.short_window} days")
        print(f"Long MA: {self.long_window} days")
        print("-"*60)
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
        print(f"Excess Return: {results['total_return'] - results['buy_hold_return']:.2f}%")
        print("-"*60)
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print("="*60)
        
        if results['trades']:
            print("\nTRADE SUMMARY:")
            print("-"*40)
            for i, trade in enumerate(results['trades'][:10], 1):  # Show first 10 trades
                print(f"{i:2d}. {trade['Date'].strftime('%Y-%m-%d')} - {trade['Action']:4s} "
                      f"${trade['Price']:8.2f} ({trade['Shares']:.2f} shares)")
            if len(results['trades']) > 10:
                print(f"... and {len(results['trades']) - 10} more trades")


def main():
    """
    Main function to demonstrate the enhanced moving average crossover strategy for US markets.
    """
    print("="*80)
    print("ENHANCED MOVING AVERAGE CROSSOVER TRADING STRATEGY - US MARKETS")
    print("="*80)
    
    # Initialize strategy with enhanced filters
    strategy = MovingAverageCrossover(
        short_window=20, 
        long_window=50,
        max_price=200,
        min_volume=1000000,
        volatility_window=20
    )
    
    print(f"Strategy Parameters:")
    print(f"  Short MA: {strategy.short_window} days")
    print(f"  Long MA: {strategy.long_window} days")
    print(f"  Max Price: ${strategy.max_price}")
    print(f"  Min Volume: {strategy.min_volume:,}")
    print(f"  Volatility Window: {strategy.volatility_window} days")
    print("="*80)
    
    # Get US market opportunities first
    print("\n1. DISCOVERING US MARKET OPPORTUNITIES...")
    opportunities = strategy.get_market_opportunities(discovery_method='comprehensive', max_stocks=300)
    
    if not opportunities.empty:
        # Use the top opportunity for backtesting
        top_stock = opportunities.iloc[0]['Symbol']
        print(f"\n2. RUNNING BACKTEST ON TOP US OPPORTUNITY: {top_stock}")
        
        try:
            # Run backtest on the top opportunity
            results = strategy.backtest(
                symbol=top_stock,
                start_date="2023-01-01",
                end_date="2024-01-01",
                initial_capital=10000
            )
            
            # Print summary
            strategy.print_summary(results)
            
            # Plot results
            strategy.plot_results(results, save_plot=True)
            
        except Exception as e:
            print(f"Error running backtest on {top_stock}: {e}")
            
            # Fallback to AAPL if top opportunity fails
            print(f"\nFalling back to AAPL...")
            try:
                results = strategy.backtest(
                    symbol="AAPL",
                    start_date="2023-01-01",
                    end_date="2024-01-01",
                    initial_capital=10000
                )
                
                strategy.print_summary(results)
                strategy.plot_results(results, save_plot=True)
                
            except Exception as e2:
                print(f"Error running backtest on AAPL: {e2}")
    else:
        print("No US market opportunities available. Running backtest on AAPL...")
        try:
            results = strategy.backtest(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2024-01-01",
                initial_capital=10000
            )
            
            strategy.print_summary(results)
            strategy.plot_results(results, save_plot=True)
            
        except Exception as e:
            print(f"Error running backtest: {e}")


if __name__ == "__main__":
    main() 