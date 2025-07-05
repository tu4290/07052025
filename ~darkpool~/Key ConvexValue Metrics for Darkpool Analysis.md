# Key ConvexValue Metrics for Darkpool Analysis

Based on the provided ConvexValue data parameters, the following metrics are most valuable for identifying and analyzing darkpool activity:

## Primary Metrics

1. **gxoi (Gamma multiplied by Open Interest)**
   - Critical for identifying key support/resistance levels
   - Large concentrations indicate price levels where market makers must hedge
   - Darkpool transactions often occur near these levels to minimize market impact

2. **dxoi (Delta multiplied by Open Interest)**
   - Reveals directional hedging pressure
   - Significant imbalances can indicate institutional positioning
   - Changes in dxoi can signal darkpool activity completion

3. **volmbs metrics (Volume of Buys minus Sells)**
   - Available across different timeframes (5m, 15m, 30m, 60m)
   - Tracks momentum and flow
   - Sudden divergences between price action and volmbs can indicate darkpool activity

4. **charmxoi (Charm multiplied by Open Interest)**
   - Measures expiration-related delta decay effects
   - Important for identifying time-sensitive darkpool positioning
   - Often correlates with end-of-day or pre-expiration darkpool activity

5. **vannaxoi (Vanna multiplied by Open Interest)**
   - Assesses volatility regime changes
   - Darkpool participants often exploit vanna effects for positioning
   - Key for identifying volatility-sensitive darkpool levels

6. **vommaxoi (Vomma multiplied by Open Interest)**
   - Measures volatility of volatility exposure
   - Important for identifying complex volatility-based darkpool strategies
   - Often indicates sophisticated institutional positioning

7. **gxvolm (Gamma multiplied by Volume)**
   - Identifies active hedging zones
   - Spikes can indicate real-time darkpool execution
   - Useful for detecting ongoing darkpool activity

8. **value_bs (Buy Value minus Sell Value)**
   - Measures directional sentiment
   - Large imbalances can indicate institutional positioning
   - Useful for confirming darkpool activity intent

## Secondary Metrics

9. **flownet**
   - (Value of Call Buys + Value of Put Sells - Value of Call Sells - Value of Put Buys)
   - Comprehensive measure of directional sentiment
   - Useful for identifying complex darkpool strategies involving both calls and puts

10. **vflowratio**
    - (Volume of Call Buys + Volume of Put Sells) / (Volume of Put Buys + Volume of Call Sells)
    - Ratio-based measure of directional sentiment
    - Useful for normalizing flow data across different market conditions

11. **put_call_ratio**
    - Traditional sentiment indicator
    - Extreme values often coincide with darkpool positioning
    - Useful as a confirmatory metric

12. **call_gxoi and put_gxoi**
    - Gamma multiplied by Open Interest for calls and puts separately
    - Allows for more granular analysis of options positioning
    - Helps identify option-type specific darkpool activity

## Timeframe-Specific Metrics

13. **volmbs_5m, volmbs_15m, volmbs_30m, volmbs_60m**
    - Volume of Buys minus Sells across different timeframes
    - Useful for identifying the timeframe of darkpool activity
    - Divergences across timeframes can indicate sophisticated darkpool execution strategies

14. **valuebs_5m, valuebs_15m, valuebs_30m, valuebs_60m**
    - Value of Buys minus Sells across different timeframes
    - Provides dollar-weighted flow information
    - Useful for identifying high-value darkpool transactions

## Call/Put Specific Metrics

15. **value_call_bs and value_put_bs**
    - Value of Buys minus Sells for calls and puts separately
    - Helps identify option-type specific darkpool positioning
    - Useful for complex darkpool strategies involving both calls and puts

16. **volm_call_bs and volm_put_bs**
    - Volume of Buys minus Sells for calls and puts separately
    - Provides contract-count flow information by option type
    - Useful for identifying option-type specific darkpool activity
