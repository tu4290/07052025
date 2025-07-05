# Elite Options Trading System - Trading Signals Documentation

## Overview of Trading Signals

The Elite Options Trading System generates various trading signals based on the calculated metrics and indicators. These signals are designed to identify potential trading opportunities across different market conditions. This document details each signal type, its calculation methodology, and its significance in the system.

## 1. Directional Signal (Bullish/Bearish)

### Description
Directional signals indicate potential price movement in a specific direction (bullish or bearish). They are primarily based on the Market Structure Position Indicator (MSPI) and related metrics.

### Calculation Logic
From `integrated_strategies_v2.py`, directional signals are generated when:
- MSPI values exceed certain thresholds
- There is alignment between MSPI and other metrics like SAI (Sentiment Alignment Indicator)
- The signal strength is determined by the number of "stars" (1-5) based on conviction level

### Parameters
- `min_directional_stars_to_issue`: Minimum stars required to issue a directional signal
- `conviction_map_high`, `conviction_map_medium`, etc.: Thresholds for different conviction levels

### Significance
- Directional signals form the basis for bullish or bearish trade ideas
- Higher star ratings indicate stronger conviction
- Used for entry timing and direction selection

## 2. SDAG Conviction Signal (Bullish/Bearish)

### Description
SDAG Conviction signals are specialized directional signals based on the alignment of multiple SDAG methodologies. They indicate high-conviction price levels where multiple calculation approaches agree.

### Calculation Logic
Generated when:
- Multiple SDAG methodologies align at a specific price level
- The alignment exceeds the `sdag_conviction_threshold_pct` threshold
- The signal direction is determined by the sign of the SDAG values

### Parameters
- `sdag_conviction_threshold_pct`: Threshold for SDAG methodology alignment
- `min_agreement_for_conviction_signal`: Minimum number of SDAG methodologies that must agree

### Significance
- SDAG Conviction signals indicate the strongest support/resistance levels
- Used for high-probability trade entries
- Often mark key reversal points in the market

## 3. Volatility Expansion Signal

### Description
Volatility Expansion signals indicate potential increases in market volatility. They are based on the Volatility Risk Indicator (VRI) and Volatility Flow Imbalance (VFI).

### Calculation Logic
Generated when:
- VRI exceeds the `vol_expansion_vri_trigger` threshold
- VFI exceeds the `vol_expansion_vfi_trigger` threshold
- The signal strength is determined by the magnitude of exceedance

### Parameters
- `vol_expansion_vri_trigger`: VRI threshold for volatility expansion
- `vol_expansion_vfi_trigger`: VFI threshold for volatility expansion
- `min_volatility_stars_to_issue`: Minimum stars required to issue a volatility signal

### Significance
- Volatility Expansion signals suggest potential for larger price movements
- Used for options strategies that benefit from increasing volatility
- Help identify when to adjust position sizing or risk management

## 4. Volatility Contraction Signal

### Description
Volatility Contraction signals indicate potential decreases in market volatility. They are based on low VRI and VFI values.

### Calculation Logic
Generated when:
- VRI falls below the `vol_contraction_vri_trigger` threshold
- VFI falls below the `vol_contraction_vfi_trigger` threshold
- The signal strength is determined by the magnitude of the shortfall

### Parameters
- `vol_contraction_vri_trigger`: VRI threshold for volatility contraction
- `vol_contraction_vfi_trigger`: VFI threshold for volatility contraction

### Significance
- Volatility Contraction signals suggest potential for smaller, more range-bound price movements
- Used for options strategies that benefit from decreasing volatility
- Help identify when to adjust position sizing or risk management

## 5. Time Decay Pin Risk Signal

### Description
Time Decay Pin Risk signals indicate potential "pinning" of price to specific strikes due to time decay effects. They are based on the Time Decay Pressure Indicator (TDPI).

### Calculation Logic
Generated when:
- TDPI exceeds the `pin_risk_tdpi_trigger` threshold
- The signal strength is determined by the magnitude of exceedance
- The signal location is at the strike with the highest TDPI

### Parameters
- `pin_risk_tdpi_trigger`: TDPI threshold for pin risk
- `min_pinrisk_stars_to_issue`: Minimum stars required to issue a pin risk signal

### Significance
- Pin Risk signals identify strikes where price might be drawn to as expiration approaches
- Used for expiration-day trading strategies
- Help identify potential profit-taking levels

## 6. Time Decay Charm Cascade Signal

### Description
Charm Cascade signals indicate potential acceleration of delta hedging due to time decay effects. They are based on the Charm Decay Rate (CTR) and Time Decay Flow Imbalance (TDFI).

### Calculation Logic
Generated when:
- CTR exceeds the `charm_cascade_ctr_trigger` threshold
- TDFI exceeds the `charm_cascade_tdfi_trigger` threshold
- The signal strength is determined by the magnitude of exceedance

### Parameters
- `charm_cascade_ctr_trigger`: CTR threshold for charm cascade
- `charm_cascade_tdfi_trigger`: TDFI threshold for charm cascade

### Significance
- Charm Cascade signals identify potential acceleration of price movement due to delta hedging
- Used for timing entries and exits around expiration
- Help identify potential volatility events

## 7. Complex Structure Change Signal

### Description
Structure Change signals indicate potential changes in market structure. They are based on the Structural Stability Index (SSI) and related metrics.

### Calculation Logic
Generated when:
- SSI falls below the `ssi_structure_change` threshold
- There are significant changes in key levels or metrics
- The signal strength is determined by the magnitude of the change

### Parameters
- `ssi_structure_change`: SSI threshold for structure change

### Significance
- Structure Change signals identify potential regime changes in the market
- Used for adjusting trading strategies to new market conditions
- Help identify when to reassess key levels and signals

## 8. Complex Flow Divergence Signal (Based on ARFI)

### Description
Flow Divergence signals indicate potential divergence between price action and options flow. They are based on the Average Relative Flow Index (ARFI) and related metrics.

### Calculation Logic
Generated when:
- ARFI exceeds the `arfi_strong_flow_threshold` or falls below the `arfi_low_flow_threshold`
- There is divergence between ARFI and price action
- The signal strength is determined by the magnitude of divergence

### Parameters
- `arfi_strong_flow_threshold`: ARFI threshold for strong flow
- `arfi_low_flow_threshold`: ARFI threshold for low flow
- `cfi_flow_divergence`: Threshold for flow divergence

### Significance
- Flow Divergence signals identify potential reversals or continuation moves
- Used for contrarian trading strategies
- Help identify when smart money might be positioning against the trend

## Signal Integration and Trade Idea Generation

### Framework Overview
The system integrates multiple signals to generate comprehensive trade ideas. The process involves:

1. **Signal Generation**: Each signal type is calculated based on its specific criteria
2. **Signal Filtering**: Signals are filtered based on minimum star thresholds
3. **Signal Integration**: Related signals are combined to form a cohesive view
4. **Trade Idea Formation**: Integrated signals are translated into actionable trade ideas
5. **Target and Stop Calculation**: Price targets and stop levels are calculated based on ATR and key levels

### Key Parameters for Trade Idea Generation
- `min_directional_stars_to_issue`: Minimum stars for directional signals
- `min_volatility_stars_to_issue`: Minimum stars for volatility signals
- `min_pinrisk_stars_to_issue`: Minimum stars for pin risk signals
- `min_caution_stars_to_issue`: Minimum stars for caution signals
- `min_reissue_time_seconds`: Minimum time between signal reissuance
- `conviction_map_*`: Thresholds for different conviction levels
- `conv_mod_*`: Conviction modifiers for different conditions

### Exit Signal Logic
The system also generates exit signals based on:
- `contradiction_stars_threshold`: Stars threshold for contradictory signals
- `ssi_exit_stars_threshold`: SSI threshold for exit signals
- `mspi_flip_threshold`: MSPI threshold for direction change
- `arfi_exit_stars_threshold`: ARFI threshold for exit signals

### Target Calculation Logic
Price targets are calculated based on:
- `min_target_atr_distance`: Minimum ATR distance for targets
- `nvp_support_quantile`, `nvp_resistance_quantile`: Quantiles for support/resistance
- `target_atr_stop_loss_multiplier`: ATR multiplier for stop loss
- `target_atr_target1_multiplier_no_sr`: ATR multiplier for first target without S/R
- `target_atr_target2_multiplier_no_sr`: ATR multiplier for second target without S/R
- `target_atr_target2_multiplier_from_t1`: ATR multiplier for second target from first target
