# merging.py

# =============================================================================
# ENHANCED MULTI-TIMEFRAME TRADING SYSTEM WITH XGBOOST & AUTOENCODER
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTS AND DEPENDENCIES
# -----------------------------------------------------------------------------
import os
import warnings
import logging
import random
import time
import pickle
import requests
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# ML Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from collections import defaultdict

# Optional dependencies with availability checks
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from gym import spaces
    from stable_baselines3 import A2C, PPO, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
    vader = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from pytrends.request import TrendReq

    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 2. CONFIGURATION AND CONSTANTS
# -----------------------------------------------------------------------------
# API Configuration
NEWSAPI_KEY = os.getenv("NEWS_API_KEY_NEWSAPI")
FRED_KEY = os.getenv("FRED_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Multi-Timeframe Configuration
TIMEFRAMES = {
    '15min': {'interval': '15min', 'time_step': 60},
    '1hour': {'interval': '1h', 'time_step': 24},
    '4hour': {'interval': '4h', 'time_step': 12},
    '1day': {'interval': '1d', 'time_step': 30}
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Trade Execution Thresholds
BUY_THRESHOLD = 0.01
SELL_THRESHOLD = -0.01

# -----------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def enhance_features(df, feature_cols):
    """Enhanced feature engineering with technical indicators"""
    try:
        enhanced_df = df.copy()
        if 'Close' in enhanced_df.columns:
            # Technical indicators
            enhanced_df['RSI'] = calculate_rsi(enhanced_df['Close'])
            macd, macd_signal = calculate_macd(enhanced_df['Close'])
            enhanced_df['MACD'] = macd
            enhanced_df['MACD_Signal'] = macd_signal
            enhanced_df['MACD_Histogram'] = macd - macd_signal
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(enhanced_df['Close'])
            enhanced_df['BB_Upper'] = bb_upper
            enhanced_df['BB_Middle'] = bb_middle
            enhanced_df['BB_Lower'] = bb_lower
            enhanced_df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            # Price-based features
            enhanced_df['Price_Change'] = enhanced_df['Close'].pct_change()
            enhanced_df['Price_Change_MA'] = enhanced_df['Price_Change'].rolling(10).mean()
            enhanced_df['Volatility'] = enhanced_df['Price_Change'].rolling(20).std()
            # Volume indicators
            if 'Volume' in enhanced_df.columns:
                enhanced_df['Volume_MA'] = enhanced_df['Volume'].rolling(20).mean()
                enhanced_df['Volume_Ratio'] = enhanced_df['Volume'] / enhanced_df['Volume_MA']
        # Fill NaN values
        enhanced_df.fillna(method='ffill', inplace=True)
        enhanced_df.fillna(0, inplace=True)
        return enhanced_df
    except Exception as e:
        logger.error(f"Error in enhance_features: {e}")
        return df

def prepare_sequence_data(df, feature_cols, time_step=60):
    """Prepare sequence data with better validation"""
    try:
        # Select only available features
        available_features = [col for col in feature_cols if col in df.columns]
        if not available_features:
            logger.error("No available features found")
            return None, None, None

        # Ensure Close price is the target
        if 'Close' not in available_features:
            logger.error("Close price not available for prediction")
            return None, None, None
        
        # Make sure Close is first feature for consistency
        if 'Close' in available_features:
            available_features.remove('Close')
            available_features.insert(0, 'Close')
        
        df = df[available_features]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
        
        if df.empty:
            logger.error("No numeric data available after filtering.")
            return None, None, None

        # Clean data first
        df = df.dropna()
        if len(df) < time_step + 10:
            logger.error("Insufficient clean data after removing NaN values")
            return None, None, None
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df.values)
        
        # Create sequences - FIXED TARGET SELECTION
        X, y = [], []
        close_index = 0  # Close should be first column now
        
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i])
            y.append(scaled_data[i, close_index])  # Always predict Close price
        
        X, y = np.array(X), np.array(y)
        print(f"Created {len(X)} sequences with time_step={time_step}")
        print(f"Target feature: {df.columns[close_index]} (Close price)")
        
        return X, y, scaler
    except Exception as e:
        logger.error(f"Error preparing sequence data: {e}")
        return None, None, None
    
def inverse_transform_prediction(pred_scaled, scaler, target_feature_index=0):
    """Properly inverse transform prediction"""
    try:
        # Create array with same shape as original features
        n_features = scaler.scale_.shape[0]
        dummy_array = np.zeros((1, n_features))
        dummy_array[0, target_feature_index] = pred_scaled
        
        # Inverse transform
        inverse_transformed = scaler.inverse_transform(dummy_array)
        return inverse_transformed[0, target_feature_index]
    except Exception as e:
        logger.error(f"Error in inverse transform: {e}")
        return pred_scaled 
    
def validate_training_data(df, feature_cols):
    """Validate training data quality"""
    print("=== Data Quality Validation ===")
    
    # Check for sufficient price variation
    if 'Close' in df.columns:
        price_std = df['Close'].std()
        price_mean = df['Close'].mean()
        cv = price_std / price_mean if price_mean > 0 else 0
        print(f"Price coefficient of variation: {cv:.4f}")
        
        if cv < 0.01:
            print("WARNING: Very low price variation - may cause prediction issues")
        
        # Check for price trends
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        print(f"Overall price change: {price_change*100:.2f}%")
        
        # Check for recent price movement
        recent_data = df.tail(30)
        recent_trend = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
        print(f"Recent 30-day trend: {recent_trend*100:.2f}%")
    
    # Check for data gaps
    missing_data = df[feature_cols].isnull().sum()
    print(f"Missing data points: {missing_data.sum()}")
    
    return True       

def save_model_checkpoint(model, ticker, model_name):
    """Save model checkpoint"""
    os.makedirs("models", exist_ok=True)
    # Sanitize the ticker for file paths
    safe_ticker = ticker.replace('/', '_')
    model_path = f"models/{safe_ticker}_{model_name}.pt"
    try:
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_path)
        else:
            with open(model_path.replace('.pt', '.pkl'), 'wb') as f:
                pickle.dump(model, f)
        logger.info(f"Model saved: {model_path}")
    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")

# -----------------------------------------------------------------------------
# 4. DATA PROVIDERS
# -----------------------------------------------------------------------------
class FMPDataProvider:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_historical_data(self, ticker, timeframe, days=365):
        """Fetch historical data from FMP API"""
        logger.info(f"Fetching data for {ticker} (timeframe: {timeframe})")
        try:
            # Sanitize ticker symbol for URL
            safe_ticker = ticker.replace('^', '%5E').replace('=', '%3D').replace('/', '%2F')

            # Define the API endpoint based on the timeframe
            if timeframe == '1d':
                # Daily historical data
                endpoint = f"https://financialmodelingprep.com/api/v3/historical-price-full/{safe_ticker}?timeseries={days}&apikey={self.api_key}"
            else:
                # Intraday historical data
                interval_map = {
                    '15min': '15min',
                    '1h': '1hour',
                    '4h': '4hour'
                }
                interval = interval_map.get(timeframe, '1hour')
                endpoint = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{safe_ticker}?apikey={self.api_key}"

            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                if 'historical' in data:
                    df = pd.DataFrame(data['historical'])
                else:
                    df = pd.DataFrame(data)
                if not df.empty:
                    # Convert date columns to datetime
                    if 'date' in df.columns:
                        df['Date'] = pd.to_datetime(df['date'])
                        df.set_index('Date', inplace=True)
                        df.drop(columns=['date'], inplace=True)
                    elif 'datetime' in df.columns:
                        df['Date'] = pd.to_datetime(df['datetime'])
                        df.set_index('Date', inplace=True)
                        df.drop(columns=['datetime'], inplace=True)
                    # Ensure columns are properly named
                    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                    return df.sort_index()
                else:
                    logger.error(f"No data returned for {ticker}")
                    return None
            else:
                logger.error(f"Failed to fetch data for {ticker}. HTTP Status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def fetch_real_time_price(self, ticker):
        """Fetch real-time price for the ticker"""
        try:
            safe_ticker = ticker.replace('^', '%5E').replace('=', '%3D').replace('/', '%2F')
            endpoint = f"https://financialmodelingprep.com/api/v3/quote-short/{safe_ticker}?apikey={self.api_key}"
            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                if data:
                    price = data[0].get('price')
                    return price
            logger.error(f"Failed to fetch real-time price for {ticker}")
            return None
        except Exception as e:
            logger.error(f"Error fetching real-time price for {ticker}: {e}")
            return None

    def _generate_sample_data(self, ticker, days):
        """Generate sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(hash(ticker) % 1000)
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, days)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, days)
        })
        return df.set_index('Date')

class MultiTimeframeDataManager:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data_cache = {}
        self.fmp_provider = FMPDataProvider(FMP_API_KEY)

    def fetch_multi_timeframe_data(self, ticker, timeframes=None):
        """Fetch data for multiple timeframes"""
        if timeframes is None:
            timeframes = list(TIMEFRAMES.keys())
        multi_tf_data = {}
        for tf in timeframes:
            try:
                data = self.fmp_provider.fetch_historical_data(
                    ticker,
                    TIMEFRAMES[tf]['interval'],
                    days=365
                )
                if data is not None and not data.empty:
                    multi_tf_data[tf] = data
                    logger.info(f"Fetched {len(data)} records for {ticker} ({tf})")
                else:
                    logger.warning(f"No data fetched for {ticker} ({tf})")
            except Exception as e:
                logger.error(f"Error fetching {tf} data for {ticker}: {e}")
        return multi_tf_data

class GoogleTrendsProvider:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360) if PYTRENDS_AVAILABLE else None

    def get_stock_trends(self, ticker, keywords=None, timeframe='today 12-m'):
        """Get Google Trends data for stock"""
        if not PYTRENDS_AVAILABLE:
            return {'trend_score': 0.5}
        try:
            if keywords is None:
                keywords = [ticker.replace('^', '').replace('=F', '')]
            # Simulate trends data
            return {
                'trend_score': np.random.uniform(0.3, 0.8),
                'trend_direction': np.random.choice(['up', 'down', 'stable'])
            }
        except Exception as e:
            logger.error(f"Error fetching trends for {ticker}: {e}")
            return {'trend_score': 0.5}

    def get_economic_trends(self, keywords=None):
        """Get economic trends data"""
        if keywords is None:
            keywords = ["recession", "inflation", "unemployment", "interest rates"]
        try:
            # Simulate economic trends
            return {
                'economic_sentiment': np.random.uniform(-0.3, 0.3),
                'economic_trend_strength': np.random.uniform(0.2, 0.8)
            }
        except Exception as e:
            logger.error(f"Error fetching economic trends: {e}")
            return {'economic_sentiment': 0.0}

class EnhancedNewsProvider:
    def __init__(self):
        self.analyzer = vader if VADER_AVAILABLE else None

    def fetch_news_sentiment(self, ticker, days_back=7):
        """Fetch news sentiment (placeholder implementation)"""
        try:
            # Simulate news sentiment
            sentiment_score = np.random.uniform(-0.5, 0.5)
            return {
                'news_sentiment_avg': sentiment_score,
                'news_volume': np.random.randint(10, 100),
                'sentiment_trend': 'positive' if sentiment_score > 0 else 'negative'
            }
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {ticker}: {e}")
            return {'news_sentiment_avg': 0.0}

class ComprehensiveDataProvider:
    def __init__(self):
        pass

    def fetch_economic_indicators(self, indicators=None):
        """Fetch economic indicators (placeholder)"""
        return {
            'gdp_growth': np.random.uniform(1.0, 4.0),
            'inflation_rate': np.random.uniform(1.5, 6.0),
            'unemployment_rate': np.random.uniform(3.0, 8.0)
        }

    def get_market_fear_indicators(self):
        """Get market fear/greed indicators"""
        return {
            'composite_fear_index': np.random.uniform(20, 80),
            'vix_level': np.random.uniform(15, 40)
        }

# -----------------------------------------------------------------------------
# 5. ANALYSIS CLASSES
# -----------------------------------------------------------------------------
class MarketRegimeDetector:
    def __init__(self, volatility_window=20, volume_threshold=1.5):
        self.volatility_window = volatility_window
        self.volume_threshold = volume_threshold

    def detect_regime(self, df):
        """Detect market regime based on volatility and volume"""
        try:
            if 'Close' not in df.columns or len(df) < self.volatility_window:
                return 'unknown'
            # Calculate volatility
            returns = df['Close'].pct_change()
            volatility = returns.rolling(self.volatility_window).std().iloc[-1]
            # Calculate volume ratio if available
            volume_ratio = 1.0
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].rolling(self.volatility_window).mean()
                volume_ratio = df['Volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
            # Determine regime
            if volatility > 0.03 and volume_ratio > self.volume_threshold:
                return 'high_volatility'
            elif volatility < 0.01:
                return 'low_volatility'
            else:
                return 'normal'
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'unknown'

class FeatureScorer:
    def __init__(self):
        pass

    def correlation_score(self, X, y):
        """Calculate correlation-based feature scores"""
        try:
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            scores = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                scores.append(abs(corr) if not np.isnan(corr) else 0)
            return np.array(scores)
        except Exception as e:
            logger.error(f"Error calculating correlation scores: {e}")
            return np.zeros(X.shape[1] if len(X.shape) == 2 else X.shape[1] * X.shape[2])

    def mutual_info_score(self, X, y):
        """Calculate mutual information scores"""
        try:
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            scores = mutual_info_regression(X, y, random_state=42)
            return scores
        except Exception as e:
            logger.error(f"Error calculating mutual info scores: {e}")
            return np.zeros(X.shape[1])

    def score_features(self, X, y, feature_names, model=None):
        """Score features using multiple methods"""
        try:
            scores = {}
            # Correlation scores
            corr_scores = self.correlation_score(X, y)
            scores['correlation'] = dict(zip(feature_names, corr_scores))
            # Mutual information scores
            mi_scores = self.mutual_info_score(X, y)
            scores['mutual_info'] = dict(zip(feature_names, mi_scores))
            return scores
        except Exception as e:
            logger.error(f"Error scoring features: {e}")
            return {}

class PredictionConfidence:
    def __init__(self):
        self.confidence_history = defaultdict(list)

    def ensemble_uncertainty(self, models, X):
        """Calculate ensemble uncertainty"""
        try:
            predictions = []
            for model in models:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions.append(pred)
            if predictions:
                predictions = np.array(predictions)
                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)
                return mean_pred, std_pred
            return None, None
        except Exception as e:
            logger.error(f"Error calculating ensemble uncertainty: {e}")
            return None, None

class MetaModelSelector:
    def __init__(self):
        self.performance_history = defaultdict(lambda: defaultdict(list))

    def update_performance(self, ticker, model_name, regime, mse, mae, sharpe=None):
        """Update model performance history"""
        self.performance_history[ticker][model_name].append({
            'regime': regime,
            'mse': mse,
            'mae': mae,
            'sharpe': sharpe,
            'timestamp': datetime.now()
        })

    def get_best_models(self, ticker, current_regime, top_k=3):
        """Get best performing models for current regime"""
        try:
            if ticker not in self.performance_history:
                return []
            model_scores = {}
            for model_name, history in self.performance_history[ticker].items():
                regime_scores = [h for h in history if h['regime'] == current_regime]
                if regime_scores:
                    avg_mse = np.mean([s['mse'] for s in regime_scores])
                    model_scores[model_name] = avg_mse
            # Sort by MSE (lower is better)
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            return [model[0] for model in sorted_models[:top_k]]
        except Exception as e:
            logger.error(f"Error getting best models: {e}")
            return []

# -----------------------------------------------------------------------------
# 6. NEURAL NETWORK MODELS
# -----------------------------------------------------------------------------
class CNNLSTMAttention(nn.Module):
    def __init__(self, n_features, seq_len=60):
        super(CNNLSTMAttention, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        # CNN layers
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        # LSTM layers
        self.lstm = nn.LSTM(128, 100, batch_first=True, dropout=0.2)
        # Attention mechanism
        self.attention = nn.MultiheadAttention(100, num_heads=4, batch_first=True)
        # Output layers
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Use last output
        x = attn_out[:, -1, :]
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last output
        x = self.output_projection(x)
        return x

class TCN(nn.Module):
    def __init__(self, n_features, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          dilation=dilation_size, padding=(kernel_size - 1) * dilation_size)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.tcn(x)
        x = x[:, :, -1]  # Use last timestep
        x = self.fc(x)
        return x

class SimpleInformer(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=2):
        super(SimpleInformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # Use last output
        x = self.output_projection(x)
        return x

class AnomalyDetectionAutoencoder(nn.Module):
    def __init__(self, input_size, seq_len, latent_dim=32, threshold_percentile=95):
        super(AnomalyDetectionAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        batch_size = x.size(0)
        # Process each time step
        encoded_seq = []
        for t in range(self.seq_len):
            encoded = self.encoder(x[:, t, :])
            encoded_seq.append(encoded)
        # Decode each time step
        decoded_seq = []
        for encoded in encoded_seq:
            decoded = self.decoder(encoded)
            decoded_seq.append(decoded)
        # Stack back to sequence
        output = torch.stack(decoded_seq, dim=1)
        return output

    def detect_anomaly(self, x):
        """Detect anomalies based on reconstruction error"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            if self.threshold is None:
                return torch.zeros_like(errors, dtype=torch.bool)
            return errors > self.threshold

    def update_threshold(self, errors):
        """Update anomaly detection threshold"""
        self.threshold = torch.quantile(errors, self.threshold_percentile / 100.0)

# -----------------------------------------------------------------------------
# 7. XGBOOST MODEL
# -----------------------------------------------------------------------------
class XGBoostTimeSeriesModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_importance_ = None

    def fit(self, X, y):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        # Reshape if needed (XGBoost expects 2D)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.feature_importance_ = self.model.feature_importances_
        return self

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        # Reshape if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

# -----------------------------------------------------------------------------
# 8. TRAINING UTILITIES
# -----------------------------------------------------------------------------
def train_model_with_validation(model, X_train, y_train, X_val, y_val, patience=15, epochs=200):
    """Train neural network model with improved loss function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use Huber loss instead of MSE for better robustness
    criterion = nn.HuberLoss(delta=0.1)  # More robust to outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    min_epochs = 20
    
    # Validate input data
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: Empty training data")
        return model.cpu()
    
    # Check target variable distribution
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    print(f"Target stats - Mean: {y_mean:.6f}, Std: {y_std:.6f}")
    
    if y_std < 1e-6:
        print("WARNING: Target variable has very low variance!")
    
    print(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # Check for NaN values
    if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
        print("Warning: NaN values detected in training data")
        return model.cpu()
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        try:
            train_pred = model(X_train_tensor)
            train_loss = criterion(train_pred.squeeze(), y_train_tensor)
            
            if torch.isnan(train_loss):
                print(f"NaN loss detected at epoch {epoch}, stopping training")
                break
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        except Exception as e:
            print(f"Training error at epoch {epoch}: {e}")
            break
        
        # Validation
        model.eval()
        with torch.no_grad():
            try:
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred.squeeze(), y_val_tensor)
                
                if torch.isnan(val_loss):
                    print(f"NaN validation loss at epoch {epoch}")
                    break
                    
            except Exception as e:
                print(f"Validation error at epoch {epoch}: {e}")
                val_loss = train_loss
        
        scheduler.step(val_loss)
        
        # Early stopping
        if epoch >= min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
                break
        
        # Progress reporting
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch}: Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    print(f"Training completed after {epoch + 1} epochs")
    return model.cpu()

# -----------------------------------------------------------------------------
# 9. MAIN TRAINING PIPELINE
# -----------------------------------------------------------------------------
def train_autoencoder_for_anomaly_detection(df, feature_cols, seq_len=30, latent_dim=32, epochs=30):
    """Train an autoencoder for anomaly detection"""
    try:
        # Prepare sequence data
        X_seq, _, _ = prepare_sequence_data(df, feature_cols, time_step=seq_len)
        if X_seq is None or len(X_seq) == 0:
            logger.error("Insufficient data to train autoencoder")
            return None, None

        # Initialize scaler
        scaler = MinMaxScaler()
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        scaled_data = scaler.fit_transform(X_flat)

        # Reshape back to (batch, seq_len, input_size)
        input_size = X_seq.shape[2]
        X_scaled = scaled_data.reshape(X_seq.shape[0], seq_len, input_size)

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Initialize model
        autoencoder = AnomalyDetectionAutoencoder(
            input_size=input_size,
            seq_len=seq_len,
            latent_dim=latent_dim
        )

        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder = autoencoder.to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print(f"Training autoencoder for {epochs} epochs...")
        for epoch in range(epochs):
            autoencoder.train()
            optimizer.zero_grad()
            reconstructed = autoencoder(X_tensor.to(device))
            loss = criterion(reconstructed, X_tensor.to(device))
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        # Update anomaly detection threshold
        autoencoder.eval()
        with torch.no_grad():
            errors = torch.mean((X_tensor - autoencoder(X_tensor)) ** 2, dim=(1, 2))
            autoencoder.update_threshold(errors)

        return autoencoder.cpu(), scaler
    except Exception as e:
        logger.error(f"Error training autoencoder: {e}")
        return None, None

# -----------------------------------------------------------------------------
# 9. MAIN TRAINING PIPELINE
# -----------------------------------------------------------------------------
def train_enhanced_models(df, feature_cols, ticker, time_step=60):
    """Train all models with enhanced debugging"""
    try:
        print(f"Input data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Validate data quality first
        validate_training_data(df, feature_cols)
        
        # Ensure Close price is available and first
        if 'Close' not in df.columns:
            print("ERROR: Close price not found in data")
            return None, None, None
        
        # Check price data
        print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"Latest price: ${df['Close'].iloc[-1]:.2f}")

        # Ensure only numeric columns are included
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.5]
        print(f"Available numeric features before filtering: {available_features}")

        # Remove any columns that are not in both feature_cols and available_features
        feature_cols = [col for col in feature_cols if col in available_features]
        print(f"Feature columns after filtering: {feature_cols}")

        if not feature_cols:
            print("No numeric features available for training after filtering.")
            return None, None, None

        # Proceed with data preparation
        X_seq, y_seq, scaler = prepare_sequence_data(df, feature_cols, time_step)
        if X_seq is None or len(X_seq) < 20:  # Need minimum sequences
            print("Insufficient sequence data for training")
            return None, None, None
        print(f"Prepared sequences: {X_seq.shape}")
        # Ensure reasonable train/validation split
        min_val_size = max(5, len(X_seq) // 10)  # At least 5 samples for validation
        split_idx = len(X_seq) - min_val_size
        X_train_seq = X_seq[:split_idx]
        X_test_seq = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        print(f"Train size: {len(X_train_seq)}, Val size: {len(X_test_seq)}")
        if len(X_train_seq) < 10:
            print("Insufficient training data for neural networks. Training with available data.")
            # Continue training with what we have - don't return early

        models = {}
        n_features = X_seq.shape[2]
        seq_len = X_seq.shape[1]
        print("Training enhanced model ensemble...")
        # 1. Train Neural Networks
        print("Training CNN-LSTM...")
        cnn_lstm = CNNLSTMAttention(n_features, seq_len)
        cnn_lstm = train_model_with_validation(
            cnn_lstm,
            X_train_seq,
            y_train,
            X_test_seq,
            y_test
        )
        models['cnn_lstm'] = cnn_lstm
        save_model_checkpoint(cnn_lstm, ticker, 'cnn_lstm')

        print("Training Transformer...")
        transformer = TimeSeriesTransformer(n_features)
        transformer = train_model_with_validation(
            transformer,
            X_train_seq,
            y_train,
            X_test_seq,
            y_test
        )
        models['transformer'] = transformer
        save_model_checkpoint(transformer, ticker, 'transformer')

        print("Training TCN...")
        tcn = TCN(n_features)
        tcn = train_model_with_validation(
            tcn,
            X_train_seq,
            y_train,
            X_test_seq,
            y_test
        )
        models['tcn'] = tcn
        save_model_checkpoint(tcn, ticker, 'tcn')

        print("Training Informer...")
        informer = SimpleInformer(n_features)
        informer = train_model_with_validation(
            informer,
            X_train_seq,
            y_train,
            X_test_seq,
            y_test
        )
        models['informer'] = informer
        save_model_checkpoint(informer, ticker, 'informer')

        # Sanitize ticker name for file paths
        safe_ticker = ticker.replace('/', '_')

        # Save the feature columns used for training
        try:
            os.makedirs("models", exist_ok=True)
            # Save the final list of features used in training
            with open(f"models/{safe_ticker}_features.pkl", "wb") as f:
                pickle.dump(feature_cols, f)
            print(f"Saved feature list for {ticker}")
        except Exception as e:
            logger.error(f"Failed to save feature list for {ticker}: {e}")

        # Save the scaler used for training
        try:
            with open(f"models/{safe_ticker}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler for {ticker}")
        except Exception as e:
            logger.error(f"Failed to save scaler for {ticker}: {e}")

        # **Added section to save training configuration**
        # ----------------------------------------------
        # Save the training configuration
        training_config = {
            'time_step': time_step,
            'feature_cols': feature_cols,
            # Add other relevant configurations as needed
        }
        try:
            with open(f"models/{safe_ticker}_config.pkl", "wb") as f:
                pickle.dump(training_config, f)
            print(f"Saved training configuration for {ticker}")
        except Exception as e:
            logger.error(f"Failed to save training configuration for {ticker}: {e}")
        # ----------------------------------------------

        # Flatten sequence data for XGBoost
        x_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        x_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

        # 2. Train XGBoost
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            xgb_model = XGBoostTimeSeriesModel(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1
            )
            xgb_model.fit(x_train_flat, y_train)
            models['xgboost'] = xgb_model
            save_model_checkpoint(xgb_model, ticker, 'xgboost')

        # 3. Train Autoencoder
        print("Training Autoencoder...")
        autoencoder_model, ae_scaler = train_autoencoder_for_anomaly_detection(
            df,
            feature_cols,
            seq_len=time_step,  # Use the same time_step
            latent_dim=32,
            epochs=30
        )
        if autoencoder_model is not None:
            models['autoencoder'] = autoencoder_model
            save_model_checkpoint(autoencoder_model, ticker, 'autoencoder')
            # Save autoencoder scaler
            try:
                with open(f"models/{safe_ticker}_autoencoder_scaler.pkl", "wb") as f:
                    pickle.dump(ae_scaler, f)
                print(f"Saved autoencoder scaler for {ticker}")
            except Exception as e:
                logger.error(f"Failed to save autoencoder scaler for {ticker}: {e}")

        # 4. Evaluate models
        print("\nModel Evaluation:")
        for eval_model_name, eval_model in models.items():
            if eval_model_name in ['ae_scaler']:
                continue
            try:
                if eval_model_name == 'xgboost':
                    pred = eval_model.predict(x_test_flat)
                    mse = mean_squared_error(y_test, pred)
                    mae = mean_absolute_error(y_test, pred)
                    print(
                        f"  {eval_model_name.upper()} - MSE: {mse:.6f}, "
                        f"MAE: {mae:.6f}"
                    )
                elif eval_model_name == 'autoencoder':
                    with torch.no_grad():
                        x_test_tensor = torch.tensor(
                            X_test_seq,
                            dtype=torch.float32
                        )
                        reconstructed = eval_model(x_test_tensor)
                        anomalies = eval_model.detect_anomaly(x_test_tensor)
                        anomaly_rate = torch.mean(anomalies.float()).item()
                        print(
                            f"  Autoencoder - Anomaly Rate: "
                            f"{anomaly_rate:.2%}"
                        )
                else:
                    eval_model.eval()
                    with torch.no_grad():
                        x_test_tensor = torch.tensor(
                            X_test_seq,
                            dtype=torch.float32
                        )
                        pred = eval_model(x_test_tensor).numpy().flatten()
                        mse = mean_squared_error(y_test, pred)
                        mae = mean_absolute_error(y_test, pred)
                        print(
                            f"  {eval_model_name.upper()} - MSE: {mse:.6f}, "
                            f"MAE: {mae:.6f}"
                        )
            except Exception as exc:
                print(f"  Error evaluating {eval_model_name}: {exc}")

        # After training, validate predictions make sense
        if len(models) > 0:
            # Test prediction on last data point
            X_seq_test, y_seq_test, scaler_test = prepare_sequence_data(df, feature_cols, time_step)
            if X_seq_test is not None and len(X_seq_test) > 0:
                X_flat_test = X_seq_test.reshape(X_seq_test.shape[0], -1)
                recent_X_seq = X_seq_test[-1:]
                recent_X_flat = X_flat_test[-1:]
                
                test_pred, _ = enhanced_ensemble_predict(models, recent_X_seq, recent_X_flat, scaler)
                if len(test_pred) > 0:
                    actual_price = df['Close'].iloc[-1]
                    predicted_price = test_pred[0]
                    print(f"\nSanity Check:")
                    print(f"Actual last price: ${actual_price:.2f}")
                    print(f"Predicted price: ${predicted_price:.2f}")
                    print(f"Difference: {((predicted_price - actual_price) / actual_price) * 100:.2f}%")
                    
                    # Flag unrealistic predictions
                    if abs(predicted_price - actual_price) > actual_price * 0.5:
                        print("WARNING: Prediction seems unrealistic!")

        # Return training_config along with models and scaler
        return models, scaler, training_config

    except Exception as exc:
        logger.error("Error in train_enhanced_models: %s", exc)
        return None, None, None

def enhanced_ensemble_predict(models_dict, x_seq, x_flat=None, scaler=None):
    """Enhanced ensemble prediction with proper scaling"""
    predictions = []
    model_names = []
    
    if x_seq is None:
        return np.array([]), []
    
    x_seq_tensor = torch.tensor(x_seq, dtype=torch.float32)
    
    for ens_model_name, ens_model in models_dict.items():
        if ens_model_name in ['ae_scaler', 'autoencoder']:
            continue
            
        try:
            if ens_model_name == 'xgboost' and x_flat is not None:
                pred_scaled = ens_model.predict(x_flat)[0]
            else:
                ens_model.eval()
                with torch.no_grad():
                    pred_scaled = ens_model(x_seq_tensor).numpy().flatten()[0]
            
            # Inverse transform the prediction
            if scaler is not None:
                pred_original = inverse_transform_prediction(pred_scaled, scaler, 0)
            else:
                pred_original = pred_scaled
            
            predictions.append(pred_original)
            model_names.append(ens_model_name)
            print(f"{ens_model_name}: scaled={pred_scaled:.6f}, original=${pred_original:.2f}")
            
        except Exception as exc:
            logger.error(f"Error getting prediction from {ens_model_name}: {exc}")
            continue
    
    if predictions:
        final_prediction = np.mean(predictions)
        print(f"Final ensemble prediction: ${final_prediction:.2f}")
        return np.array([final_prediction]), model_names
    
    return np.array([]), []

# -----------------------------------------------------------------------------
# 10. MANAGEMENT CLASSES (PLACEHOLDERS)
# -----------------------------------------------------------------------------
class ModelRetrainingManager:
    def __init__(self):
        self.retrain_schedule = {}

    def schedule_retrain(self, ticker, interval_hours=24):
        """Schedule model retraining"""
        self.retrain_schedule[ticker] = interval_hours
        logger.info(f"Scheduled retraining for {ticker} every {interval_hours} hours")

class AutomatedTradingScheduler:
    def __init__(self, tickers, prediction_engine, retrain_manager):
        self.tickers = tickers
        self.prediction_engine = prediction_engine
        self.retrain_manager = retrain_manager
        self.running = False

    def start_scheduler(self, max_runs=1):  # Add max_runs parameter
        """Start the automated scheduler"""
        self.running = True
        run_count = 0
        logger.info("Automated trading scheduler started")
        while self.running and run_count < max_runs:  # Add run limit
            for ticker in self.tickers:
                try:
                    # Fetch latest data
                    data_manager = MultiTimeframeDataManager([ticker])
                    data = data_manager.fetch_multi_timeframe_data(ticker).get('1day')
                    if data is None or data.empty:
                        continue
                    # Make prediction
                    prediction = self.prediction_engine.make_realtime_prediction(ticker, data)
                    print(f"Prediction for {ticker}: {prediction}")
                    # Execute trade
                    if prediction is not None:
                        execute_trade(ticker, prediction)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
            run_count += 1
            time.sleep(60)  # Wait before next iteration
        self.stop_scheduler()

    def stop_scheduler(self):
        """Stop the automated scheduler"""
        self.running = False
        logger.info("Automated trading scheduler stopped")

class ModelVersionManager:
    def __init__(self):
        self.versions = defaultdict(list)

    def save_model_version(self, ticker, model_name, model, metrics):
        """Save model version with metadata"""
        # Sanitize the ticker for file paths
        safe_ticker = ticker.replace('/', '_')
        version_info = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'model_path': f"models/{safe_ticker}_{model_name}_v{len(self.versions[ticker])}.pt"
        }
        self.versions[ticker].append(version_info)
        logger.info(f"Saved model version for {safe_ticker}_{model_name}")

class RealTimePredictionEngine:
    def __init__(self, tickers):
        self.tickers = tickers
        self.models = {}
        self.configs = {}

    def load_models(self, ticker):
        """Load models and configurations for ticker"""
        loaded_models = {}
        # Sanitize the ticker for file paths
        safe_ticker = ticker.replace('/', '_')
        config_path = f"models/{safe_ticker}_config.pkl"
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                training_config = pickle.load(f)
            n_features = len(training_config['feature_cols'])
            seq_len = training_config['time_step']
        else:
            logger.error(f"Configuration file not found for {ticker}")
            return None
        model_names = ['cnn_lstm', 'transformer', 'tcn', 'informer', 'xgboost']
        for model_name in model_names:
            model_path = f"models/{safe_ticker}_{model_name}.pt"
            if os.path.exists(model_path) or os.path.exists(model_path.replace('.pt', '.pkl')):
                if model_name == 'cnn_lstm':
                    model = CNNLSTMAttention(n_features, seq_len)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                elif model_name == 'transformer':
                    model = TimeSeriesTransformer(n_features)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                elif model_name == 'tcn':
                    model = TCN(n_features)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                elif model_name == 'informer':
                    model = SimpleInformer(n_features)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                elif model_name == 'xgboost':
                    # Load XGBoost model
                    with open(model_path.replace('.pt', '.pkl'), 'rb') as f:
                        model = pickle.load(f)
                loaded_models[model_name] = model
        return loaded_models

    def make_realtime_prediction(self, ticker, current_data):
        """Make real-time prediction"""
        models = self.load_models(ticker)
        if models is None:
            return None
        # Sanitize the ticker for file paths
        safe_ticker = ticker.replace('/', '_')
        # Load configuration
        config_path = f"models/{safe_ticker}_config.pkl"
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found for {ticker}")
            return None
        with open(config_path, "rb") as f:
            training_config = pickle.load(f)
        time_step = training_config['time_step']
        feature_cols = training_config['feature_cols']
        
        # Load scaler
        scaler_path = f"models/{safe_ticker}_scaler.pkl"
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        
        # Feature engineering
        enhanced_df = enhance_features(current_data, feature_cols)
        # Prepare sequence data
        X_seq, _, _ = prepare_sequence_data(enhanced_df, feature_cols, time_step=time_step)
        if X_seq is not None and len(X_seq) > 0:
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            recent_X_seq = X_seq[-1:]
            recent_X_flat = X_flat[-1:]
            ensemble_pred, _ = enhanced_ensemble_predict(models, recent_X_seq, recent_X_flat, scaler)
            return ensemble_pred[0] if len(ensemble_pred) > 0 else None
        return None

# -----------------------------------------------------------------------------
# 11. TRADE EXECUTION LOGIC
# -----------------------------------------------------------------------------
def execute_trade(ticker, prediction, threshold=0.01):
    """Execute trade based on prediction"""
    try:
        if prediction > threshold:
            print(f"Buy signal for {ticker}: {prediction:.4f}")
            # Place buy order via trading API
            # Example: trading_api.buy(ticker, quantity=1)
        elif prediction < -threshold:
            print(f"Sell signal for {ticker}: {prediction:.4f}")
            # Place sell order via trading API
            # Example: trading_api.sell(ticker, quantity=1)
        else:
            print(f"No trade for {ticker}: {prediction:.4f}")
    except Exception as e:
        logger.error(f"Error executing trade for {ticker}: {e}")

# -----------------------------------------------------------------------------
# 12. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    tickers = ["^GDAXI", "GC=F", "KC=F", "NG=F", "CC=F", "^HSI"]

    # Initialize ALL data providers FIRST
    print("Initializing data providers...")
    data_manager = MultiTimeframeDataManager(tickers)
    trends_provider = GoogleTrendsProvider()
    news_provider = EnhancedNewsProvider()
    comprehensive_data = ComprehensiveDataProvider()

    # Initialize analysis tools
    regime_detector = MarketRegimeDetector()
    feature_scorer = FeatureScorer()
    meta_selector = MetaModelSelector()
    confidence_estimator = PredictionConfidence()

    # Initialize management components
    retrain_manager = ModelRetrainingManager()
    model_version_manager = ModelVersionManager()
    prediction_engine = RealTimePredictionEngine(tickers)
    scheduler = AutomatedTradingScheduler(tickers, prediction_engine, retrain_manager)

    # Start scheduler
    print("Running model training pipeline...")

    try:
        for ticker in tickers:
            try:
                print(f"\n{'='*60}")
                print(f"PROCESSING TICKER: {ticker}")
                print(f"{'='*60}")

                # Fetch data
                multi_tf_data = data_manager.fetch_multi_timeframe_data(ticker)
                if not multi_tf_data:
                    print(f"No data available for {ticker}")
                    continue

                # Use daily data for processing
                data = multi_tf_data.get('1day', next(iter(multi_tf_data.values())))
                if data.empty:
                    print(f"Empty data for {ticker}")
                    continue

                # Market regime detection
                current_regime = regime_detector.detect_regime(data)
                print(f"Current market regime: {current_regime}")

                # Feature engineering
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                enhanced_df = enhance_features(data, feature_cols)
                if enhanced_df is None or enhanced_df.empty:
                    print(f"Failed to enhance features for {ticker}")
                    continue

                # Get available features
                available_features = [col for col in enhanced_df.columns
                                    if enhanced_df[col].notna().sum() > len(enhanced_df) * 0.5]
                print(f"Available features: {len(available_features)}")

                # External data collection
                print("\nCollecting external data...")
                external_data = {}

                # Google Trends
                trends_data = trends_provider.get_stock_trends(ticker)
                if trends_data:
                    external_data.update(trends_data)
                    print(f"Google Trends score: {trends_data.get('trend_score', 'N/A')}")

                # Economic trends
                economic_trends = trends_provider.get_economic_trends()
                if economic_trends:
                    external_data.update(economic_trends)

                # News sentiment
                news_sentiment = news_provider.fetch_news_sentiment(ticker)
                if news_sentiment:
                    external_data.update(news_sentiment)
                    print(f"News sentiment: {news_sentiment.get('news_sentiment_avg', 'N/A'):.3f}")

                # Economic indicators
                economic_data = comprehensive_data.fetch_economic_indicators()
                fear_data = comprehensive_data.get_market_fear_indicators()
                if fear_data:
                    external_data.update({f"fear_{k}": v for k, v in fear_data.items()})
                    print(f"Market fear index: {fear_data.get('composite_fear_index', 'N/A'):.1f}")

                # Train models
                print(f"\nTraining models for {ticker}...")
                models, scaler, training_config = train_enhanced_models(enhanced_df, available_features, ticker=ticker)
                if models:
                    print(f"Successfully trained {len([k for k in models.keys() if k != 'ae_scaler'])} models")

                    # Save models
                    for model_name, model in models.items():
                        if model_name not in ['ae_scaler']:
                            save_model_checkpoint(model, ticker, model_name)

                    # Make predictions
                    if len(enhanced_df) > 30:
                        X_seq, y_seq, _ = prepare_sequence_data(enhanced_df, available_features, 30)
                        if X_seq is not None and len(X_seq) > 0:
                            X_flat = X_seq.reshape(X_seq.shape[0], -1)
                            recent_X_seq = X_seq[-1:]
                            recent_X_flat = X_flat[-1:]
                            ensemble_pred, used_models = enhanced_ensemble_predict(
                                models, recent_X_seq, recent_X_flat, scaler
                            )
                            if len(ensemble_pred) > 0:
                                print(f"\nEnsemble prediction: {ensemble_pred[0]:.4f}")
                                print(f"Models used: {used_models}")

                                # Anomaly detection
                                if 'autoencoder' in models:
                                    autoencoder = models['autoencoder']
                                    is_anomaly = autoencoder.detect_anomaly(
                                        torch.tensor(recent_X_seq, dtype=torch.float32)
                                    )
                                    print(f"Anomaly detected: {is_anomaly.item()}")

                print(f"Completed processing for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue  # Continue with the next ticker

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
    finally:
        scheduler.stop_scheduler()

    print("All models trained and saved successfully!")
    
    # Model validation
    print("\n" + "="*80)
    print("MODEL VALIDATION")
    print("="*80)
    
    for ticker in tickers:
        try:
            safe_ticker = ticker.replace('/', '_')
            config_path = f"models/{safe_ticker}_config.pkl"
            
            if os.path.exists(config_path):
                # Load data and make prediction
                multi_tf_data = data_manager.fetch_multi_timeframe_data(ticker)
                data = multi_tf_data.get('1day', next(iter(multi_tf_data.values()))) if multi_tf_data else None
                
                if data is not None and not data.empty:
                    actual_price = data['Close'].iloc[-1]
                    predicted_price = prediction_engine.make_realtime_prediction(ticker, data)
                    
                    if predicted_price is not None:
                        price_diff_pct = ((predicted_price - actual_price) / actual_price) * 100
                        print(f"{ticker}: Actual=${actual_price:.2f}, Predicted=${predicted_price:.2f}, Diff={price_diff_pct:+.2f}%")
                        
                        if abs(price_diff_pct) > 50:
                            print(f"  ⚠️  WARNING: Unrealistic prediction for {ticker}")
                    else:
                        print(f"{ticker}: Prediction failed")
        except Exception as e:
            print(f"{ticker}: Validation error - {e}")