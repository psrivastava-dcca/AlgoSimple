import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// Interfaces for the trading data
export interface MarketOpportunity {
  symbol: string;
  price: number;
  volume: string;
  volatility: number;
  maTrend: number;
  score: number;
  recommendation: string;
}

export interface MarketSummary {
  totalOpportunities: number;
  strongBuy: number;
  buy: number;
  weakBuy: number;
  timestamp: string;
}

export interface MarketDiscoveryResponse {
  success: boolean;
  summary: MarketSummary;
  opportunities: MarketOpportunity[];
  rawOutput: string;
  timestamp: string;
}

export interface BacktestRequest {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
}

export interface BacktestResults {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  buy_hold_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  num_trades: number;
  trades: any[];
}

export interface BacktestResponse {
  success: boolean;
  results: BacktestResults;
  timestamp: string;
}

@Injectable({
  providedIn: 'root'
})
export class TradingService {
  private apiUrl = 'http://localhost:5050/api';

  constructor(private http: HttpClient) { }

  healthCheck(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health`);
  }

  getMarketDiscovery(maxStocks: number = 50, minPrice: number = 0, maxPrice: number = 200): Observable<MarketDiscoveryResponse> {
    return this.http.get<MarketDiscoveryResponse>(`${this.apiUrl}/market-discovery`, {
      params: {
        max_stocks: maxStocks.toString(),
        min_price: minPrice.toString(),
        max_price: maxPrice.toString()
      }
    });
  }

  runBacktest(request: BacktestRequest): Observable<BacktestResponse> {
    return this.http.post<BacktestResponse>(`${this.apiUrl}/backtest`, request);
  }
}
