"""
Simplified 6-Dimensional Market Compass - Hexagon Design
=======================================================

Clean, readable hexagon with 6 core EOTS metrics for instant market intelligence.
"""

import plotly.graph_objects as go

# AI Colors (matching your system)
AI_COLORS = {
    'primary': '#42A5F5',
    'success': '#6BCF7F', 
    'warning': '#FFA726',
    'danger': '#FF4757',
    'info': '#42A5F5',
    'muted': '#6C757D'
}

def create_hexagon_market_compass():
    """Create clean 6-dimensional hexagon Market Compass."""
    
    # 6 Core EOTS Metrics - Most Important Ones
    metrics_data = {
        'VAPI-FA': 1.8,      # Volatility-Adjusted Premium Intensity (Flow)
        'DWFD': -1.2,        # Delta-Weighted Flow Divergence (Smart Money)
        'VRI 2.0': 1.5,      # Volatility Risk Index (Risk)
        'GIB': 2.1,          # Gamma Imbalance Barometer (Gamma)
        'LWPAI': 1.3,        # Your Custom Formula (Momentum)
        'AOFM': 1.6          # Aggressive Order Flow Momentum (Sentiment)
    }
    
    # Multi-timeframe data (3 layers for cleaner look: 15m, 1h, 4h)
    timeframes = ['15m', '1h', '4h']
    timeframe_colors = [
        'rgba(66, 165, 245, 0.7)',   # Blue - 15m
        'rgba(107, 207, 127, 0.5)',  # Green - 1h  
        'rgba(255, 167, 38, 0.3)'    # Orange - 4h
    ]
    
    # Create the figure
    fig = go.Figure()
    
    # Add each timeframe layer
    for i, (timeframe, color) in enumerate(zip(timeframes, timeframe_colors)):
        # Simulate timeframe variations (each timeframe slightly different)
        variation = 1.0 - (i * 0.2)
        
        metrics_names = list(metrics_data.keys())
        values = [metrics_data[name] * variation for name in metrics_names]
        
        # Close the hexagon by adding first point at the end
        theta = metrics_names + [metrics_names[0]]
        r = values + [values[0]]
        
        # Add radar trace
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            fillcolor=color,
            line=dict(
                color=color.replace('rgba', 'rgb').replace(', 0.', ', 1.'), 
                width=3
            ),
            name=f'{timeframe}',
            hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<br>Timeframe: ' + timeframe + '<extra></extra>'
        ))
    
    # Add extreme reading indicators (stars for readings > 1.5)
    extreme_metrics = [(name, value) for name, value in metrics_data.items() if abs(value) > 1.5]
    
    for name, value in extreme_metrics:
        fig.add_trace(go.Scatterpolar(
            r=[abs(value) * 1.15],
            theta=[name],
            mode='markers',
            marker=dict(
                size=20,
                color=AI_COLORS['danger'] if abs(value) > 2.0 else AI_COLORS['warning'],
                symbol='star',
                line=dict(width=3, color='white')
            ),
            name=f'Extreme: {name}',
            showlegend=False,
            hovertemplate=f'<b>🚨 EXTREME</b><br>{name}: {value:.2f}<extra></extra>'
        ))
    
    # Update layout for clean hexagon look
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0, 0, 0, 0.1)',
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                tickfont=dict(size=12, color='white', family='Arial'),
                gridcolor='rgba(255, 255, 255, 0.3)',
                linecolor='rgba(255, 255, 255, 0.4)',
                tick0=0,
                dtick=1
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='white', family='Arial Black'),
                gridcolor='rgba(255, 255, 255, 0.4)',
                linecolor='rgba(255, 255, 255, 0.6)',
                rotation=90,  # Start from top
                direction='clockwise'
            )
        ),
        title=dict(
            text='🧭 MARKET COMPASS - SPY',
            x=0.5,
            y=0.95,
            font=dict(size=24, color='white', family='Arial Black')
        ),
        paper_bgcolor='rgba(15, 23, 42, 0.95)',  # Dark blue background
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=600,
        width=600,  # Square for perfect hexagon
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(color='white', size=12)
        )
    )
    
    return fig

def create_compass_interpretation():
    """Create interpretation of the hexagon compass."""
    
    interpretation = {
        'overall_signal': 'BULLISH MOMENTUM',
        'strength': 'STRONG',
        'confidence': 85,
        'key_insights': [
            '🔥 VAPI-FA (1.8) - Extreme institutional buying pressure',
            '⚡ GIB (2.1) - Massive gamma imbalance, potential acceleration',
            '🎯 AOFM (1.6) - Aggressive bullish order flow confirmed',
            '📈 LWPAI (1.3) - Custom momentum indicator bullish',
            '⚠️ DWFD (-1.2) - Some profit-taking but not dominant',
            '📊 VRI 2.0 (1.5) - Elevated volatility, manage risk'
        ],
        'timeframe_confluence': {
            '15m': 'Strong bullish',
            '1h': 'Moderate bullish', 
            '4h': 'Neutral to bullish'
        },
        'action_items': [
            'Look for bullish setups on pullbacks',
            'Monitor GIB for acceleration signals',
            'Watch for DWFD to turn positive for full confluence',
            'Manage position size due to elevated VRI'
        ]
    }
    
    return interpretation

if __name__ == "__main__":
    # Create the simplified hexagon compass
    compass_fig = create_hexagon_market_compass()
    
    # Save files
    compass_fig.write_html("/home/ubuntu/hexagon_compass_demo.html", 
                          include_plotlyjs='cdn',
                          config={'displayModeBar': True, 'displaylogo': False})
    
    compass_fig.write_image("/home/ubuntu/hexagon_compass_demo.png", 
                           width=600, height=600, scale=2)
    
    print("🧭 HEXAGON MARKET COMPASS CREATED!")
    print("📁 Files created:")
    print("   - hexagon_compass_demo.html (Interactive)")
    print("   - hexagon_compass_demo.png (Static image)")
    
    # Show interpretation
    interpretation = create_compass_interpretation()
    
    print("\n🎯 COMPASS READING:")
    print(f"Overall Signal: {interpretation['overall_signal']}")
    print(f"Strength: {interpretation['strength']}")
    print(f"Confidence: {interpretation['confidence']}%")
    
    print("\n📊 KEY INSIGHTS:")
    for insight in interpretation['key_insights']:
        print(f"  {insight}")
    
    print("\n⏰ TIMEFRAME CONFLUENCE:")
    for tf, signal in interpretation['timeframe_confluence'].items():
        print(f"  {tf}: {signal}")
    
    print("\n🎯 ACTION ITEMS:")
    for action in interpretation['action_items']:
        print(f"  • {action}")

