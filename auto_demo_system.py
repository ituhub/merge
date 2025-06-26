# auto_demo_system.py
"""
Advanced Auto-Demo Generation System for AI Trading Dashboard
Provides automated demo content generation, screen capture simulation, and trading simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import json
import os
from datetime import datetime, timedelta
import random

class AutoDemoGenerator:
    """Automated demo content generator for AI Trading Dashboard"""
    
    def __init__(self):
        self.demo_active = False
        self.demo_thread = None
        self.demo_data = {}
        self.current_step = 0
        self.demo_scenarios = self.load_demo_scenarios()
        
    def load_demo_scenarios(self):
        """Load predefined demo scenarios"""
        return {
            "bullish_trend": {
                "name": "🚀 Bullish Market Trend",
                "duration": 60,  # seconds
                "tickers": ["^GDAXI", "GC=F", "^HSI"],
                "price_changes": [2.5, 1.8, 3.2],
                "signals": ["BUY", "BUY", "BUY"],
                "descriptions": [
                    "Strong upward momentum detected",
                    "Gold showing safe-haven strength", 
                    "Asian markets leading global rally"
                ]
            },
            "volatile_market": {
                "name": "⚡ High Volatility Trading",
                "duration": 90,
                "tickers": ["KC=F", "NG=F", "CC=F"],
                "price_changes": [-2.1, 4.5, -1.8],
                "signals": ["SELL", "BUY", "SELL"],
                "descriptions": [
                    "Coffee futures showing weakness",
                    "Natural gas surge on supply concerns",
                    "Cocoa facing demand pressures"
                ]
            },
            "diversified_portfolio": {
                "name": "📊 Multi-Asset Strategy",
                "duration": 120,
                "tickers": ["^GDAXI", "GC=F", "KC=F", "NG=F"],
                "price_changes": [1.2, -0.8, 2.3, -1.5],
                "signals": ["BUY", "HOLD", "BUY", "SELL"],
                "descriptions": [
                    "German index showing stability",
                    "Gold consolidating in range",
                    "Coffee breaking resistance",
                    "Natural gas profit-taking"
                ]
            },
            "breakout_strategy": {
                "name": "💥 Breakout Trading",
                "duration": 75,
                "tickers": ["^GDAXI", "KC=F", "CC=F"],
                "price_changes": [4.2, -3.1, 2.8],
                "signals": ["STRONG_BUY", "STRONG_SELL", "BUY"],
                "descriptions": [
                    "DAX breaking major resistance",
                    "Coffee major breakdown signal",
                    "Cocoa momentum building"
                ]
            },
            "risk_management": {
                "name": "🛡️ Risk Management Demo",
                "duration": 100,
                "tickers": ["GC=F", "NG=F", "^HSI", "CC=F"],
                "price_changes": [-2.5, 1.2, -1.8, 3.4],
                "signals": ["SELL", "HOLD", "SELL", "BUY"],
                "descriptions": [
                    "Gold triggering stop loss",
                    "Natural gas in consolidation",
                    "HSI showing weakness",
                    "Cocoa recovery signal"
                ]
            }
        }
    
    def start_auto_demo(self, scenario_name="bullish_trend"):
        """Start automated demo with specified scenario"""
        if self.demo_active:
            return False, "Demo already running"
        
        if scenario_name not in self.demo_scenarios:
            return False, f"Scenario '{scenario_name}' not found"
        
        self.demo_active = True
        self.current_step = 0
        self.demo_data = self.demo_scenarios[scenario_name].copy()
        
        # Start background thread for demo
        self.demo_thread = threading.Thread(target=self._run_demo_sequence)
        self.demo_thread.daemon = True
        self.demo_thread.start()
        
        return True, f"Auto-demo started: {self.demo_data['name']}"
    
    def stop_auto_demo(self):
        """Stop the automated demo"""
        self.demo_active = False
        if self.demo_thread:
            self.demo_thread.join(timeout=1)
        return "Auto-demo stopped"
    
    def _run_demo_sequence(self):
        """Internal method to run demo sequence"""
        scenario = self.demo_data
        tickers = scenario['tickers']
        duration_per_ticker = scenario['duration'] / len(tickers)
        
        for i, ticker in enumerate(tickers):
            if not self.demo_active:
                break
                
            # Update session state for current ticker
            st.session_state['demo_current_ticker'] = ticker
            st.session_state['demo_current_step'] = i + 1
            st.session_state['demo_total_steps'] = len(tickers)
            st.session_state['demo_description'] = scenario['descriptions'][i]
            st.session_state['demo_signal'] = scenario['signals'][i]
            st.session_state['demo_price_change'] = scenario['price_changes'][i]
            st.session_state['demo_scenario_name'] = scenario['name']
            
            # Simulate ticker switch
            st.session_state['selected_ticker'] = ticker
            
            # Wait for duration
            time.sleep(duration_per_ticker)
        
        # Demo completed
        self.demo_active = False
        st.session_state['demo_completed'] = True
    
    def get_demo_status(self):
        """Get current demo status"""
        return {
            'active': self.demo_active,
            'current_step': st.session_state.get('demo_current_step', 0),
            'total_steps': st.session_state.get('demo_total_steps', 0),
            'current_ticker': st.session_state.get('demo_current_ticker', ''),
            'description': st.session_state.get('demo_description', ''),
            'signal': st.session_state.get('demo_signal', ''),
            'price_change': st.session_state.get('demo_price_change', 0),
            'scenario_name': st.session_state.get('demo_scenario_name', '')
        }

class AutoScreenCapture:
    """Automated screen capture simulation during demo"""
    
    def __init__(self):
        self.capturing = False
        self.screenshots = []
        self.capture_interval = 5  # seconds
        
    def start_capture(self):
        """Start automated screenshot capture simulation"""
        self.capturing = True
        self.screenshots = []
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        
        return "Screenshot capture started"
    
    def stop_capture(self):
        """Stop screenshot capture"""
        self.capturing = False
        return f"Captured {len(self.screenshots)} screenshots"
    
    def _capture_loop(self):
        """Internal capture loop"""
        while self.capturing:
            try:
                # Simulate screenshot data
                screenshot_data = {
                    'timestamp': datetime.now().isoformat(),
                    'ticker': st.session_state.get('selected_ticker', ''),
                    'demo_step': st.session_state.get('demo_current_step', 0),
                    'signal': st.session_state.get('demo_signal', ''),
                    'price_change': st.session_state.get('demo_price_change', 0),
                    'description': st.session_state.get('demo_description', '')
                }
                self.screenshots.append(screenshot_data)
                
                # Wait for next capture
                time.sleep(self.capture_interval)
                
            except Exception as e:
                print(f"Screenshot capture error: {e}")
                break
    
    def export_screenshots_log(self):
        """Export screenshots log as JSON"""
        return json.dumps(self.screenshots, indent=2)
    
    def get_capture_summary(self):
        """Get summary of captured screenshots"""
        if not self.screenshots:
            return {}
        
        summary = {
            'total_captures': len(self.screenshots),
            'duration': (datetime.fromisoformat(self.screenshots[-1]['timestamp']) - 
                        datetime.fromisoformat(self.screenshots[0]['timestamp'])).total_seconds(),
            'tickers_captured': list(set(s['ticker'] for s in self.screenshots)),
            'signals_captured': list(set(s['signal'] for s in self.screenshots))
        }
        return summary

class AutoTradingSimulator:
    """Simulate automated trading during demo"""
    
    def __init__(self):
        self.auto_trading_active = False
        self.simulated_trades = []
        
    def start_auto_trading_demo(self):
        """Start automated trading simulation"""
        self.auto_trading_active = True
        self.simulated_trades = []
        
        # Start trading thread
        trading_thread = threading.Thread(target=self._simulate_trading)
        trading_thread.daemon = True
        trading_thread.start()
        
        return "Auto-trading simulation started"
    
    def stop_auto_trading_demo(self):
        """Stop automated trading simulation"""
        self.auto_trading_active = False
        return f"Simulated {len(self.simulated_trades)} trades"
    
    def _simulate_trading(self):
        """Simulate trading based on demo signals"""
        while self.auto_trading_active:
            try:
                # Get current demo status
                current_signal = st.session_state.get('demo_signal', 'HOLD')
                current_ticker = st.session_state.get('demo_current_ticker', '')
                price_change = st.session_state.get('demo_price_change', 0)
                
                if current_signal in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL'] and abs(price_change) > 1.0:
                    # Simulate trade execution
                    trade = {
                        'timestamp': datetime.now().isoformat(),
                        'ticker': current_ticker,
                        'action': current_signal,
                        'signal_strength': abs(price_change) / 5.0,
                        'simulated': True,
                        'expected_return': price_change,
                        'risk_level': 'High' if abs(price_change) > 3 else 'Medium' if abs(price_change) > 2 else 'Low'
                    }
                    
                    self.simulated_trades.append(trade)
                    
                    # Add to demo portfolio if exists
                    if 'portfolio' in st.session_state:
                        self._execute_simulated_trade(trade)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Auto-trading simulation error: {e}")
                break
    
    def _execute_simulated_trade(self, trade):
        """Execute simulated trade in demo portfolio"""
        portfolio = st.session_state['portfolio']
        
        # Simulate current price
        base_prices = {
            '^GDAXI': 23400, 'GC=F': 2050, 'KC=F': 185,
            'NG=F': 3.25, 'CC=F': 245, '^HSI': 19500
        }
        
        ticker = trade['ticker']
        base_price = base_prices.get(ticker, 1000)
        price_change = st.session_state.get('demo_price_change', 0)
        current_price = base_price * (1 + price_change / 100)
        
        if trade['action'] in ['BUY', 'STRONG_BUY'] and portfolio['cash'] > 1000:
            # Simulate buy
            amount = min(portfolio['cash'] * 0.2, 2000)  # 20% or $2000 max
            if trade['action'] == 'STRONG_BUY':
                amount = min(portfolio['cash'] * 0.3, 3000)  # 30% or $3000 max
            
            shares = amount / current_price
            
            portfolio['cash'] -= amount
            portfolio['positions'][ticker] = portfolio['positions'].get(ticker, 0) + shares
            
            # Add to trade history
            trade_record = {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'action': trade['action'],
                'shares': shares,
                'price': current_price,
                'amount': amount,
                'signal_strength': trade['signal_strength'],
                'reason': f"🤖 Auto-demo {trade['action'].lower()} ({price_change:+.2f}%)"
            }
            portfolio['trade_history'].append(trade_record)
            
        elif trade['action'] in ['SELL', 'STRONG_SELL']:
            # Simulate sell
            current_position = portfolio['positions'].get(ticker, 0)
            if current_position > 0:
                sell_ratio = 0.5 if trade['action'] == 'SELL' else 0.8  # Sell more on strong signal
                shares_to_sell = current_position * sell_ratio
                proceeds = shares_to_sell * current_price
                
                portfolio['cash'] += proceeds
                portfolio['positions'][ticker] = current_position - shares_to_sell
                
                # Add to trade history
                trade_record = {
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': trade['action'],
                    'shares': shares_to_sell,
                    'price': current_price,
                    'amount': proceeds,
                    'signal_strength': trade['signal_strength'],
                    'reason': f"🤖 Auto-demo {trade['action'].lower()} ({price_change:+.2f}%)"
                }
                portfolio['trade_history'].append(trade_record)
    
    def get_trading_summary(self):
        """Get summary of simulated trading"""
        if not self.simulated_trades:
            return {}
        
        buy_trades = [t for t in self.simulated_trades if 'BUY' in t['action']]
        sell_trades = [t for t in self.simulated_trades if 'SELL' in t['action']]
        
        summary = {
            'total_trades': len(self.simulated_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'avg_signal_strength': np.mean([t['signal_strength'] for t in self.simulated_trades]),
            'tickers_traded': list(set(t['ticker'] for t in self.simulated_trades)),
            'strong_signals': len([t for t in self.simulated_trades if 'STRONG' in t['action']])
        }
        return summary

class DemoVideoRecorder:
    """Video recording simulation for demo sessions"""
    
    def __init__(self):
        self.recording = False
        self.video_sessions = []
        self.current_session = None
        
    def start_recording(self, session_name=None):
        """Start video recording session"""
        if self.recording:
            return False, "Recording already active"
        
        if not session_name:
            session_name = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.recording = True
        self.current_session = {
            'name': session_name,
            'start_time': datetime.now(),
            'frames': [],
            'events': []
        }
        
        return True, f"Recording started: {session_name}"
    
    def stop_recording(self):
        """Stop video recording session"""
        if not self.recording:
            return False, "No active recording"
        
        self.recording = False
        if self.current_session:
            self.current_session['end_time'] = datetime.now()
            self.current_session['duration'] = (
                self.current_session['end_time'] - self.current_session['start_time']
            ).total_seconds()
            self.video_sessions.append(self.current_session)
            
        return True, f"Recording saved: {self.current_session['name']}"
    
    def add_event(self, event_type, description):
        """Add event to current recording session"""
        if self.recording and self.current_session:
            event = {
                'timestamp': datetime.now(),
                'type': event_type,
                'description': description
            }
            self.current_session['events'].append(event)
    
    def get_recording_status(self):
        """Get current recording status"""
        if not self.recording:
            return {'active': False, 'session': None}
        
        duration = (datetime.now() - self.current_session['start_time']).total_seconds()
        return {
            'active': True,
            'session': self.current_session['name'],
            'duration': duration,
            'events_count': len(self.current_session['events'])
        }

# Main Demo Control Interface Functions
def setup_auto_demo_controls():
    """Setup automated demo controls in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎬 **Auto-Demo Generator**")
    
    # Initialize demo components
    if 'auto_demo' not in st.session_state:
        st.session_state['auto_demo'] = AutoDemoGenerator()
    
    if 'auto_capture' not in st.session_state:
        st.session_state['auto_capture'] = AutoScreenCapture()
    
    if 'auto_trading_sim' not in st.session_state:
        st.session_state['auto_trading_sim'] = AutoTradingSimulator()
    
    if 'video_recorder' not in st.session_state:
        st.session_state['video_recorder'] = DemoVideoRecorder()
    
    demo_gen = st.session_state['auto_demo']
    auto_capture = st.session_state['auto_capture']
    auto_trading = st.session_state['auto_trading_sim']
    video_recorder = st.session_state['video_recorder']
    
    # Demo scenario selection
    scenario_options = list(demo_gen.demo_scenarios.keys())
    scenario_names = [demo_gen.demo_scenarios[key]['name'] for key in scenario_options]
    
    selected_scenario_idx = st.sidebar.selectbox(
        "📋 Demo Scenario",
        range(len(scenario_options)),
        format_func=lambda x: scenario_names[x],
        help="Choose demo scenario to run"
    )
    selected_scenario = scenario_options[selected_scenario_idx]
    
    # Show scenario details
    scenario_info = demo_gen.demo_scenarios[selected_scenario]
    st.sidebar.info(f"""
    **{scenario_info['name']}**
    
    Duration: {scenario_info['duration']}s
    Assets: {len(scenario_info['tickers'])}
    Signals: {', '.join(set(scenario_info['signals']))}
    """)
    
    # Demo controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if not demo_gen.demo_active:
            if st.button("🎬 Start Auto-Demo", help="Start automated demo sequence"):
                success, message = demo_gen.start_auto_demo(selected_scenario)
                if success:
                    st.sidebar.success(message)
                    
                    # Auto-start capture and trading simulation
                    auto_capture.start_capture()
                    auto_trading.start_auto_trading_demo()
                    video_recorder.start_recording()
                else:
                    st.sidebar.error(message)
        else:
            if st.button("⏹️ Stop Demo", help="Stop automated demo"):
                demo_gen.stop_auto_demo()
                auto_capture.stop_capture()
                auto_trading.stop_auto_trading_demo()
                video_recorder.stop_recording()
                st.sidebar.info("Auto-demo stopped")
    
    with col2:
        # Demo status
        demo_status = demo_gen.get_demo_status()
        if demo_status['active']:
            progress = demo_status['current_step'] / max(demo_status['total_steps'], 1)
            st.sidebar.progress(progress)
            st.sidebar.write(f"Step {demo_status['current_step']}/{demo_status['total_steps']}")
        else:
            st.sidebar.info("Demo Ready")
    
    # Advanced demo options
    with st.sidebar.expander("🔧 Advanced Demo Options"):
        # Auto-screenshot settings
        capture_interval = st.slider("Screenshot Interval (sec)", 1, 30, 5)
        auto_capture.capture_interval = capture_interval
        
        # Auto-trading settings
        enable_auto_trading = st.checkbox("🤖 Enable Auto-Trading Simulation", value=True)
        
        # Demo speed multiplier
        demo_speed = st.selectbox("⚡ Demo Speed", [0.5, 1.0, 2.0, 5.0], index=1)
        
        # Video recording settings
        enable_video_recording = st.checkbox("📹 Enable Video Recording", value=True)
        
        # Export options
        if st.button("📥 Export Demo Data"):
            export_demo_data(auto_capture, auto_trading, video_recorder)
    
    # Show current demo status
    if demo_status['active']:
        st.sidebar.markdown("### 🎯 **Live Demo Status**")
        st.sidebar.success(f"🎬 **DEMO ACTIVE**")
        st.sidebar.info(f"📊 Current: {demo_status['current_ticker']}")
        st.sidebar.write(f"📝 {demo_status['description']}")
        
        # Signal indicator
        signal = demo_status['signal']
        signal_color = {"BUY": "🟢", "SELL": "🔴", "STRONG_BUY": "💚", "STRONG_SELL": "❤️", "HOLD": "🟡"}.get(signal, "⚪")
        st.sidebar.write(f"🎯 Signal: {signal_color} **{signal}**")
        st.sidebar.write(f"📈 Change: {demo_status['price_change']:+.2f}%")
        
        # Recording status
        recording_status = video_recorder.get_recording_status()
        if recording_status['active']:
            st.sidebar.write(f"📹 Recording: {recording_status['duration']:.0f}s")

def display_auto_demo_dashboard():
    """Display auto-demo dashboard when active"""
    if 'auto_demo' not in st.session_state:
        return
    
    demo_gen = st.session_state['auto_demo']
    demo_status = demo_gen.get_demo_status()
    
    if not demo_status['active']:
        return
    
    # Demo header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #FF6B6B, #4ECDC4); 
                padding: 20px; border-radius: 15px; margin: 20px 0; 
                color: white; text-align: center;">
        <h2>🎬 LIVE AUTO-DEMO ACTIVE</h2>
        <h3>{demo_status['scenario_name']}</h3>
        <p>Automated demo sequence running - AI predictions updating in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo progress
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Demo Step", 
            f"{demo_status['current_step']}/{demo_status['total_steps']}"
        )
    
    with col2:
        st.metric(
            "Current Asset", 
            demo_status['current_ticker']
        )
    
    with col3:
        signal = demo_status['signal']
        signal_color = {"BUY": "🟢", "SELL": "🔴", "STRONG_BUY": "💚", "STRONG_SELL": "❤️", "HOLD": "🟡"}.get(signal, "⚪")
        st.metric(
            "AI Signal", 
            f"{signal_color} {signal}"
        )
    
    with col4:
        st.metric(
            "Expected Move", 
            f"{demo_status['price_change']:+.2f}%"
        )
    
    with col5:
        # Demo timer (approximate)
        if demo_status['active']:
            st.metric("Status", "🔴 LIVE")
        else:
            st.metric("Status", "⏸️ PAUSED")
    
    # Demo description
    st.info(f"📝 **Current Analysis**: {demo_status['description']}")
    
    # Auto-trading status
    if 'auto_trading_sim' in st.session_state:
        auto_trading = st.session_state['auto_trading_sim']
        if auto_trading.auto_trading_active:
            trading_summary = auto_trading.get_trading_summary()
            st.success(f"🤖 **Auto-Trading Simulation Active** - {trading_summary.get('total_trades', 0)} trades executed")

def display_demo_completed_summary():
    """Display summary when demo is completed"""
    if not st.session_state.get('demo_completed', False):
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                padding: 30px; border-radius: 20px; margin: 20px 0; 
                color: white; text-align: center;">
        <h2>✅ AUTO-DEMO COMPLETED</h2>
        <p>Automated demo sequence finished successfully!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo statistics
    col1, col2, col3 = st.columns(3)
    
    # Trading statistics
    if 'auto_trading_sim' in st.session_state:
        auto_trading = st.session_state['auto_trading_sim']
        trading_summary = auto_trading.get_trading_summary()
        
        with col1:
            st.markdown("### 🤖 Trading Summary")
            st.metric("Simulated Trades", trading_summary.get('total_trades', 0))
            st.metric("Buy Signals", trading_summary.get('buy_trades', 0))
            st.metric("Sell Signals", trading_summary.get('sell_trades', 0))
    
    # Capture statistics
    if 'auto_capture' in st.session_state:
        auto_capture = st.session_state['auto_capture']
        capture_summary = auto_capture.get_capture_summary()
        
        with col2:
            st.markdown("### 📸 Capture Summary")
            st.metric("Screenshots", capture_summary.get('total_captures', 0))
            st.metric("Duration", f"{capture_summary.get('duration', 0):.0f}s")
            st.metric("Assets Captured", len(capture_summary.get('tickers_captured', [])))
    
    # Video recording statistics
    if 'video_recorder' in st.session_state:
        video_recorder = st.session_state['video_recorder']
        
        with col3:
            st.markdown("### 📹 Recording Summary")
            st.metric("Video Sessions", len(video_recorder.video_sessions))
            if video_recorder.video_sessions:
                last_session = video_recorder.video_sessions[-1]
                st.metric("Last Duration", f"{last_session.get('duration', 0):.0f}s")
                st.metric("Events Recorded", len(last_session.get('events', [])))
    
    # Show detailed trade summary if available
    if 'auto_trading_sim' in st.session_state:
        auto_trading = st.session_state['auto_trading_sim']
        if auto_trading.simulated_trades:
            st.markdown("### 📈 Simulated Trade Details")
            trade_df = pd.DataFrame([
                {
                    'Time': pd.to_datetime(trade['timestamp']).strftime('%H:%M:%S'),
                    'Ticker': trade['ticker'],
                    'Action': f"{'🟢' if 'BUY' in trade['action'] else '🔴'} {trade['action']}",
                    'Expected Return': f"{trade['expected_return']:+.2f}%",
                    'Risk Level': trade['risk_level'],
                    'Signal Strength': f"{trade['signal_strength']:.1%}",
                    'Type': '🤖 Auto-Demo'
                }
                for trade in auto_trading.simulated_trades
            ])
            st.dataframe(trade_df, use_container_width=True)
    
    # Reset demo completed flag
    if st.button("🔄 Reset Demo Status"):
        st.session_state['demo_completed'] = False
        st.rerun()

def export_demo_data(auto_capture, auto_trading, video_recorder):
    """Export all demo data for analysis"""
    demo_data = {
        'export_timestamp': datetime.now().isoformat(),
        'screenshots': auto_capture.screenshots,
        'trading_simulation': auto_trading.simulated_trades,
        'video_sessions': video_recorder.video_sessions,
        'capture_summary': auto_capture.get_capture_summary(),
        'trading_summary': auto_trading.get_trading_summary()
    }
    
    # Create download button
    export_json = json.dumps(demo_data, indent=2, default=str)
    st.download_button(
        "💾 Download Demo Data",
        data=export_json,
        file_name=f"demo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("Demo data prepared for download!")

# Utility Functions
def get_demo_ticker_info():
    """Get ticker information for demo display"""
    return {
        "^GDAXI": {"name": "DAX", "flag": "🇩🇪", "type": "Index", "country": "Germany"},
        "GC=F": {"name": "Gold", "flag": "🥇", "type": "Commodity", "country": "Global"},
        "KC=F": {"name": "Coffee", "flag": "☕", "type": "Commodity", "country": "Global"},
        "NG=F": {"name": "Natural Gas", "flag": "⛽", "type": "Energy", "country": "Global"},
        "CC=F": {"name": "Cocoa", "flag": "🍫", "type": "Commodity", "country": "Global"},
        "^HSI": {"name": "Hang Seng", "flag": "🇭🇰", "type": "Index", "country": "Hong Kong"}
    }

def simulate_market_conditions():
    """Simulate various market conditions for demo"""
    conditions = {
        'volatile': {'multiplier': 1.5, 'description': 'High volatility conditions'},
        'stable': {'multiplier': 0.7, 'description': 'Stable market conditions'},
        'trending': {'multiplier': 1.2, 'description': 'Strong trending market'},
        'sideways': {'multiplier': 0.5, 'description': 'Sideways market movement'}
    }
    
    return random.choice(list(conditions.items()))
