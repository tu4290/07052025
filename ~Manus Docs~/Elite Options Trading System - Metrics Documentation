# Elite Options Trading System - Metrics Documentation

## Core Metrics Overview

The Elite Options Trading System utilizes a sophisticated set of metrics and indicators to analyze options market structure and generate trading signals. This document provides a comprehensive overview of each metric, its calculation methodology, and its role in the system.

## 1. Delta Adjusted Gamma Exposure (DAG - Custom)

### Description
DAG combines delta and gamma exposure to provide a more comprehensive view of market structure than either metric alone. It reveals where delta and gamma forces align or conflict, creating stronger signals at key price levels.

### Calculation
From `integrated_strategies_v2.py`, the DAG calculation uses:
```
df['dag_custom'] = (
    gamma_exposure_values * dxoi_sign *
    (1 + df['alpha'] * df['net_delta_flow_to_dxoi_ratio']) *
    df['norm_net_gamma_flow']
)
```

Where:
- `gamma_exposure_values`: Values from the gamma exposure source column (typically "gxoi")
- `dxoi_sign`: Sign of the delta exposure (positive or negative)
- `alpha`: Coefficient based on alignment between delta flow and delta OI
- `net_delta_flow_to_dxoi_ratio`: Ratio of net delta flow to delta OI
- `norm_net_gamma_flow`: Normalized net gamma flow

### Parameters
- `gamma_exposure_source_col`: Configurable source column for gamma exposure (default: "gxoi")
- `delta_exposure_source_col`: Configurable source column for delta exposure (default: "dxoi")
- `direct_delta_buy_col`: Column for direct delta buy flow (default: "deltas_buy")
- `direct_delta_sell_col`: Column for direct delta sell flow (default: "deltas_sell")
- `direct_gamma_buy_col`: Column for direct gamma buy flow (default: "gammas_buy")
- `direct_gamma_sell_col`: Column for direct gamma sell flow (default: "gammas_sell")
- `proxy_delta_flow_col`: Proxy column for delta flow (default: "dxvolm")
- `proxy_gamma_flow_col`: Proxy column for gamma flow (default: "gxvolm")

### Significance
- Strong positive DAG indicates stability with directional bias in that direction
- Alignment of all DAG methodologies at specific price levels provides higher conviction signals
- Used as a component in the Market Structure Position Indicator (MSPI)

## 2. Skew and Delta Adjusted GEX (SDAG) Methodologies

### Description
SDAG enhances the DAG concept by incorporating volatility skew effects with delta exposure. It provides a more nuanced view of market structure by accounting for volatility skew across different strikes.

### Methodologies

#### 2.1 SDAG Multiplicative
```
SDAG = Skew_Adjusted_GEX * (1 + Delta_Weight)
```

#### 2.2 SDAG Directional
```
SDAG = Skew_Adjusted_GEX * sign(Skew_Adjusted_GEX * Delta) * (1 + |Delta|)
```

#### 2.3 SDAG Weighted
```
SDAG = (w1 * Skew_Adjusted_GEX + w2 * Delta) / (w1 + w2)
```

#### 2.4 SDAG Volatility-Focused
```
SDAG = Skew_Adjusted_GEX * (1 + Delta * sign(Skew_Adjusted_GEX))
```

### Parameters
- `use_skew_adjusted_for_sdag`: Whether to use skew-adjusted gamma (default: false)
- `skew_adjusted_gamma_source_col`: Source column for skew-adjusted gamma (default: "sgxoi")
- Configuration for each methodology in `dag_methodologies` section of config

### Significance
- SDAG values above 1.5 indicate extremely strong positive signals (strong support/resistance)
- SDAG values below -1.5 indicate extremely strong negative signals (volatility triggers)
- Alignment of all four SDAG methodologies provides highest conviction trading signals
- For day trading, focus on Multiplicative and Directional values
- For swing trading, focus on Weighted values
- For options strategies, focus on Volatility values

## 3. Time Decay Pressure Indicator (TDPI)

### Description
TDPI measures the pressure from time decay (theta) and its rate of change (charm) on option positions. It helps identify areas where time decay effects are most pronounced.

### Calculation
From `integrated_strategies_v2.py`:
```
df['tdpi'] = (
    charmxoi_numeric * txoi_sign *
    (1 + df['beta'] * df['charm_flow_to_charm_oi_ratio']) *
    df['norm_net_theta_flow'] *
    df['time_weight'] *
    df['strike_proximity']
)
```

Where:
- `charmxoi_numeric`: Charm multiplied by open interest
- `txoi_sign`: Sign of theta exposure
- `beta`: Coefficient based on alignment between charm flow and charm OI
- `charm_flow_to_charm_oi_ratio`: Ratio of charm flow to charm OI
- `norm_net_theta_flow`: Normalized net theta flow
- `time_weight`: Weight based on time of day
- `strike_proximity`: Proximity to current price based on ATR

### Related Metrics
- **Charm Decay Rate (CTR)**: Ratio of charm flow to theta flow
- **Time Decay Flow Imbalance (TDFI)**: Ratio of normalized theta flow to normalized theta OI

### Parameters
- `direct_theta_buy_col`: Column for direct theta buy flow (default: "thetas_buy")
- `direct_theta_sell_col`: Column for direct theta sell flow (default: "thetas_sell")
- `proxy_theta_flow_col`: Proxy column for theta flow (default: "txvolm")
- `proxy_charm_flow_col`: Proxy column for charm flow (default: "charmxvolm")

### Significance
- High TDPI values indicate areas where time decay effects are strongest
- Used to identify potential "pin risk" areas where price might be drawn to specific strikes
- CTR and TDFI help identify potential "charm cascade" events where delta hedging accelerates

## 4. Volatility Risk Indicator (VRI)

### Description
VRI measures the risk associated with changes in implied volatility. It combines vanna (delta sensitivity to volatility changes) and vega (option price sensitivity to volatility changes) to identify areas where volatility changes have the most impact.

### Calculation
From `integrated_strategies_v2.py`:
```
df['vri'] = (
    vannaxoi_numeric * vxoi_sign *
    (1 + df['gamma_v'] * df['vanna_flow_to_vanna_oi_ratio']) *
    df['norm_net_vega_flow'] *
    df['vol_context_weight'] *
    df['vomma_factor']
)
```

Where:
- `vannaxoi_numeric`: Vanna multiplied by open interest
- `vxoi_sign`: Sign of vega exposure
- `gamma_v`: Coefficient based on alignment between vanna flow and vanna OI
- `vanna_flow_to_vanna_oi_ratio`: Ratio of vanna flow to vanna OI
- `norm_net_vega_flow`: Normalized net vega flow
- `vol_context_weight`: Weight based on current IV vs historical IV
- `vomma_factor`: Factor based on vomma (volatility of volatility)

### Related Metrics
- **Volatility Volume Ratio (VVR)**: Ratio of vanna flow to vega flow
- **Volatility Flow Imbalance (VFI)**: Ratio of normalized vega flow to normalized vega OI

### Parameters
- `direct_vega_buy_col`: Column for direct vega buy flow (default: "vegas_buy")
- `direct_vega_sell_col`: Column for direct vega sell flow (default: "vegas_sell")
- `proxy_vega_flow_col`: Proxy column for vega flow (default: "vxvolm")
- `proxy_vanna_flow_col`: Proxy column for vanna flow (default: "vannaxvolm")
- `proxy_vomma_flow_col`: Proxy column for vomma flow (default: "vommaxvolm")

### Significance
- High VRI values indicate areas where volatility changes have the most impact
- Used to identify potential volatility expansion or contraction zones
- VVR and VFI help identify potential volatility regime changes

## 5. Market Structure Position Indicator (MSPI)

### Description
MSPI is the core composite indicator that integrates multiple metrics to provide a comprehensive view of market structure. It combines DAG, TDPI, VRI, and SDAG methodologies with configurable weights.

### Calculation
The MSPI is a weighted sum of normalized component metrics:
```
MSPI = w1*DAG + w2*TDPI + w3*VRI + w4*SDAG_Multiplicative + w5*SDAG_Weighted + w6*SDAG_Volatility_Focused
```

Where weights are configured based on time of day or volatility regime.

### Parameters
- Weights for each component configured in `data_processor_settings.weights` section
- Time-based weight profiles for morning, midday, and final trading sessions
- Volatility-based weight profiles for high and low IV environments

### Significance
- MSPI provides a comprehensive view of market structure at each price level
- Used to identify key support and resistance levels
- Forms the basis for trading signals and recommendations

## 6. Sentiment Alignment Indicator (SAI)

### Description
SAI measures the alignment of sentiment across different market participants and timeframes. It helps identify when sentiment is uniformly bullish or bearish.

### Calculation
SAI is calculated based on the alignment of various flow metrics:
```
SAI = f(net_volume_pressure, net_value_pressure, net_delta_flow, net_gamma_flow)
```

### Parameters
- `sai_high_conviction` threshold in strategy settings

### Significance
- High SAI values indicate strong sentiment alignment
- Used to identify high-conviction trading opportunities
- Component in identifying key levels

## 7. Structural Stability Index (SSI)

### Description
SSI measures the stability of market structure over time. It helps identify when market structure is changing or remaining stable.

### Calculation
SSI is calculated based on the consistency of key levels and metrics over time.

### Parameters
- `ssi_structure_change` threshold in strategy settings
- `ssi_vol_contraction` threshold in strategy settings
- `ssi_conviction_split` threshold in strategy settings

### Significance
- High SSI values indicate stable market structure
- Low SSI values indicate potential structure changes
- Used to identify potential trading opportunities during structure changes

## 8. Average Relative Flow Index (ARFI)

### Description
ARFI measures the relative flow of options activity compared to historical averages. It helps identify unusual activity that might signal directional moves.

### Parameters
- `arfi_strong_flow_threshold` in strategy settings
- `arfi_low_flow_threshold` in strategy settings

### Significance
- High ARFI values indicate unusually strong options flow
- Used to identify potential directional moves
- Component in complex flow divergence signals

## 9. Order Flow Imbalance (OFI)

### Description
OFI measures the imbalance between buy and sell orders. It helps identify potential directional pressure.

### Calculation
OFI is calculated based on the ratio of buy to sell volume:
```
OFI = volm_buy / volm_sell
```

### Significance
- High OFI values indicate strong buying pressure
- Low OFI values indicate strong selling pressure
- Input to other metrics and signals

## 10. Volatility Flow Imbalance (VFI)

### Description
VFI measures the imbalance between vega buying and selling. It helps identify potential volatility regime changes.

### Calculation
VFI is calculated based on the ratio of vega buying to vega selling:
```
VFI = vegas_buy / vegas_sell
```

### Parameters
- `vol_expansion_vfi_trigger` in strategy settings
- `vol_contraction_vfi_trigger` in strategy settings

### Significance
- High VFI values indicate potential volatility expansion
- Low VFI values indicate potential volatility contraction
- Used in volatility expansion and contraction signals

## 11. Charm Decay Rate (CTR) & Time Decay Flow Imbalance (TDFI)

### Description
CTR measures the rate of charm decay, while TDFI measures the imbalance between theta buying and selling. Together, they help identify potential "charm cascade" events.

### Calculation
```
CTR = charm_flow / theta_flow
TDFI = normalized_theta_flow / normalized_theta_oi
```

### Parameters
- `charm_cascade_ctr_trigger` in strategy settings
- `charm_cascade_tdfi_trigger` in strategy settings

### Significance
- High CTR and TDFI values indicate potential charm cascade events
- Used in time decay charm cascade signals
- Help identify when delta hedging might accelerate due to time decay
