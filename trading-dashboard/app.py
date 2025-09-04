from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import sys
import os
import json
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add the parent directory to Python path to import the strategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Trading Dashboard API is running'
    })

@app.route('/api/market-discovery', methods=['GET'])
def market_discovery():
    """Run market discovery and return results"""
    try:
        # Get parameters from query string
        max_stocks = request.args.get('max_stocks', 50, type=int)
        min_price = request.args.get('min_price', 0, type=float)
        max_price = request.args.get('max_price', 200, type=float)
        
        # Path to the test_market_discovery.py script (in the same directory as app.py)
        script_path = os.path.join(os.path.dirname(__file__), 'test_market_discovery.py')
        
        # Run the market discovery script
        result = subprocess.run([
            sys.executable, 
            script_path
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            return jsonify({
                'error': 'Failed to run market discovery',
                'stderr': result.stderr,
                'return_code': result.returncode
            }), 500
        
        # Parse the output to extract structured data
        output_lines = result.stdout.strip().split('\n')
        
        # Extract opportunities from the output
        opportunities = []
        in_opportunities_section = False
        
        for line in output_lines:
            line = line.strip()
            
            # Look for the opportunities section
            if 'TOP US MARKET OPPORTUNITIES:' in line:
                in_opportunities_section = True
                continue
                
            if in_opportunities_section and line.startswith('Found'):
                break
                
            if in_opportunities_section and '|' in line and '$' in line:
                # Parse opportunity line
                try:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 6:
                        symbol = parts[0].strip()
                        price_str = parts[1].strip().replace('$', '').strip()
                        volume_str = parts[2].strip().replace('Vol:', '').strip()
                        volatility_str = parts[3].strip().replace('Volatility:', '').strip().replace('%', '')
                        ma_trend_str = parts[4].strip().replace('MA Trend:', '').strip().replace('%', '')
                        score_str = parts[5].strip().replace('Score:', '').strip()
                        
                        # Clean up the strings
                        price = float(price_str) if price_str.replace('.', '').replace('-', '').isdigit() else 0
                        volume = volume_str.replace(',', '') if volume_str else '0'
                        volatility = float(volatility_str) if volatility_str.replace('.', '').replace('-', '').isdigit() else 0
                        ma_trend = float(ma_trend_str) if ma_trend_str.replace('.', '').replace('-', '').isdigit() else 0
                        score = int(score_str) if score_str.isdigit() else 0
                        
                        # Apply price range filtering
                        if min_price <= price <= max_price:
                            opportunities.append({
                                'symbol': symbol,
                                'price': price,
                                'volume': volume,
                                'volatility': volatility,
                                'maTrend': ma_trend,
                                'score': score,
                                'recommendation': 'STRONG BUY' if score == 5 else 'BUY' if score >= 3 else 'WEAK BUY'
                            })
                except Exception as e:
                    continue
        
        # Extract summary statistics
        summary = {
            'totalOpportunities': len(opportunities),
            'strongBuy': len([o for o in opportunities if o['score'] == 5]),
            'buy': len([o for o in opportunities if o['score'] >= 3 and o['score'] < 5]),
            'weakBuy': len([o for o in opportunities if o['score'] < 3]),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'opportunities': opportunities[:max_stocks],
            'rawOutput': result.stdout,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run a backtest for a specific symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-01-01')
        initial_capital = data.get('initial_capital', 10000)
        
        # Import the strategy class
        from moving_average_crossover import MovingAverageCrossover
        
        # Initialize strategy
        strategy = MovingAverageCrossover(
            short_window=20,
            long_window=50,
            max_price=200,
            min_volume=1000000,
            volatility_window=20
        )
        
        # Run backtest
        results = strategy.backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # Extract key metrics
        backtest_results = {
            'symbol': results['symbol'],
            'startDate': results['start_date'],
            'endDate': results['end_date'],
            'initialCapital': results['initial_capital'],
            'finalValue': results['final_value'],
            'totalReturn': results['total_return'],
            'buyHoldReturn': results['buy_hold_return'],
            'excessReturn': results['total_return'] - results['buy_hold_return'],
            'sharpeRatio': results['sharpe_ratio'],
            'maxDrawdown': results['max_drawdown'],
            'numTrades': results['num_trades'],
            'trades': results['trades']
        }
        
        return jsonify({
            'success': True,
            'results': backtest_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("Starting Trading Dashboard API...")
    print("API will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    print("Market discovery: http://localhost:5000/api/market-discovery")
    app.run(debug=True, host='0.0.0.0', port=5000)
