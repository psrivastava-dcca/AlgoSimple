import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TradingService } from '../../services/trading.service';
import { MarketOpportunity, MarketSummary } from '../../services/trading.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  opportunities: MarketOpportunity[] = [];
  summary: MarketSummary | null = null;
  loading = false;
  error: string | null = null;
  maxStocks = 50;
  minPrice = 0;
  maxPrice = 200;

  constructor(private tradingService: TradingService) {}

  ngOnInit() {
    this.refreshData();
  }

  refreshData() {
    // Validate price range
    if (this.minPrice >= this.maxPrice) {
      this.error = 'Minimum price must be less than maximum price';
      return;
    }
    
    this.loading = true;
    this.error = null;
    
    this.tradingService.getMarketDiscovery(this.maxStocks, this.minPrice, this.maxPrice).subscribe({
      next: (response) => {
        this.opportunities = response.opportunities;
        this.summary = response.summary;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load market data: ' + err.message;
        this.loading = false;
      }
    });
  }

  getRecommendationClass(recommendation: string): string {
    switch (recommendation) {
      case 'STRONG BUY': return 'strong-buy';
      case 'BUY': return 'buy';
      case 'WEAK BUY': return 'weak-buy';
      case 'HOLD': return 'hold';
      default: return 'neutral';
    }
  }

  formatVolume(volume: string): string {
    const num = parseInt(volume);
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return volume;
  }

  setPriceRange(min: number, max: number) {
    this.minPrice = min;
    this.maxPrice = max;
    this.refreshData();
  }
}
