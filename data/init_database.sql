-- Elite Options System Database Schema
-- Initialize tables for options data, portfolio tracking, and analytics

-- Options Chain Data Table
CREATE TABLE IF NOT EXISTS options_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    expiration_date DATE NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    last_price DECIMAL(10,4),
    volume INTEGER DEFAULT 0,
    open_interest INTEGER DEFAULT 0,
    implied_volatility DECIMAL(8,4),
    delta DECIMAL(8,4),
    gamma DECIMAL(8,4),
    theta DECIMAL(8,4),
    vega DECIMAL(8,4),
    rho DECIMAL(8,4),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, expiration_date, strike_price, option_type, timestamp)
);

-- Portfolio Positions Table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    position_type VARCHAR(10) NOT NULL CHECK (position_type IN ('LONG', 'SHORT')),
    option_type VARCHAR(4) CHECK (option_type IN ('CALL', 'PUT', 'STOCK')),
    strike_price DECIMAL(10,2),
    expiration_date DATE,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    unrealized_pnl DECIMAL(12,2),
    realized_pnl DECIMAL(12,2) DEFAULT 0,
    entry_date DATETIME NOT NULL,
    exit_date DATETIME,
    status VARCHAR(10) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Historical Performance Data
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    total_portfolio_value DECIMAL(15,2),
    daily_pnl DECIMAL(12,2),
    cumulative_pnl DECIMAL(15,2),
    win_rate DECIMAL(5,2),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    volatility DECIMAL(8,4),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Market Data Cache
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    volume INTEGER,
    market_cap BIGINT,
    pe_ratio DECIMAL(8,2),
    dividend_yield DECIMAL(5,2),
    beta DECIMAL(6,3),
    fifty_two_week_high DECIMAL(10,4),
    fifty_two_week_low DECIMAL(10,4),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

-- Trading Signals and Alerts
CREATE TABLE IF NOT EXISTS trading_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    signal_strength DECIMAL(3,2) CHECK (signal_strength BETWEEN 0 AND 1),
    description TEXT,
    trigger_price DECIMAL(10,4),
    target_price DECIMAL(10,4),
    stop_loss DECIMAL(10,4),
    expiration_date DATETIME,
    status VARCHAR(10) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'TRIGGERED', 'EXPIRED', 'CANCELLED')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    triggered_at DATETIME
);

-- Risk Metrics Table
CREATE TABLE IF NOT EXISTS risk_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    portfolio_beta DECIMAL(6,3),
    portfolio_delta DECIMAL(10,2),
    portfolio_gamma DECIMAL(10,2),
    portfolio_theta DECIMAL(10,2),
    portfolio_vega DECIMAL(10,2),
    var_95 DECIMAL(12,2),
    var_99 DECIMAL(12,2),
    expected_shortfall DECIMAL(12,2),
    concentration_risk DECIMAL(5,2),
    leverage_ratio DECIMAL(6,3),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_options_symbol_exp ON options_chains(symbol, expiration_date);
CREATE INDEX IF NOT EXISTS idx_options_timestamp ON options_chains(timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_status ON portfolio_positions(status);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_status ON trading_signals(status);
CREATE INDEX IF NOT EXISTS idx_risk_date ON risk_metrics(date);

-- Insert some sample data for testing
INSERT OR IGNORE INTO market_data (symbol, price, volume, market_cap, pe_ratio, beta) VALUES
('SPY', 445.50, 50000000, 400000000000, 18.5, 1.0),
('QQQ', 375.25, 35000000, 200000000000, 25.2, 1.15),
('AAPL', 185.75, 45000000, 2800000000000, 28.8, 1.25),
('MSFT', 375.80, 25000000, 2750000000000, 32.1, 0.95),
('GOOGL', 135.40, 20000000, 1650000000000, 22.7, 1.05);

INSERT OR IGNORE INTO performance_metrics (date, total_portfolio_value, daily_pnl, cumulative_pnl, win_rate, total_trades) VALUES
('2024-01-15', 100000.00, 0.00, 0.00, 0.00, 0),
('2024-01-16', 101250.00, 1250.00, 1250.00, 100.00, 1),
('2024-01-17', 102100.00, 850.00, 2100.00, 100.00, 2);