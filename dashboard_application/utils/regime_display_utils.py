"""
Utility functions for converting internal regime names to user-friendly tactical display names.
This module provides centralized regime name mapping for consistent UI display across the system.
"""

def get_tactical_regime_name(internal_regime_name: str) -> str:
    """
    PYDANTIC-FIRST: Convert internal regime names to user-friendly tactical display names.

    Args:
        internal_regime_name: The internal regime name from the system

    Returns:
        User-friendly tactical display name
    """
    # PYDANTIC-FIRST: Handle None/empty values with proper defaults
    # ENHANCED DIAGNOSTIC: Log what regime value we're getting
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 REGIME DISPLAY: Received regime value: '{internal_regime_name}' (type: {type(internal_regime_name)})")

    if not internal_regime_name or internal_regime_name in [None, "None", "UNKNOWN", ""]:
        logger.warning(f"🚨 REGIME DISPLAY: Invalid regime value, showing 'Analyzing...' - Value was: '{internal_regime_name}'")
        return "Unknown Regime: Analyzing..."

    regime_display_mapping = {
        "REGIME_SPX_0DTE_FRIDAY_EOD_VANNA_CASCADE_POTENTIAL_BULLISH": "Vanna Squeeze: Bullish Cascade",
        "REGIME_SPY_PRE_FOMC_VOL_COMPRESSION_WITH_DWFD_ACCUMULATION": "Apex Ambush: Smart Money Loading",
        "REGIME_HIGH_VAPI_FA_BULLISH_MOMENTUM_UNIVERSAL": "Alpha Surge: Bullish Conviction",
        "REGIME_ADAPTIVE_STRUCTURE_BREAKDOWN_WITH_DWFD_CONFIRMATION_BEARISH_UNIVERSAL": "Structure Breach: Bearish Confirmed",
        "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH": "Ignition Point: Volatile Up",
        "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH": "Ignition Point: Volatile Down",
        "REGIME_NVP_STRONG_BUY_IMBALANCE_AT_KEY_STRIKE": "Demand Wall: Flow Support",
        "REGIME_EOD_HEDGING_PRESSURE_BUY": "Closing Imbalance: Systemic Buying",
        "REGIME_EOD_HEDGING_PRESSURE_SELL": "Closing Imbalance: Systemic Selling",
        "REGIME_SIDEWAYS_MARKET": "Consolidation: Tactical Chop",
        "REGIME_HIGH_VOLATILITY": "Chaos State: Elevated Risk",
        "REGIME_UNCLEAR_OR_TRANSITIONING": "Transition State: Standby"
    }

    # Return tactical name if found, otherwise fallback to formatted internal name
    try:
        return regime_display_mapping.get(internal_regime_name, internal_regime_name.replace("_", " ").title())
    except AttributeError:
        # Handle case where internal_regime_name is not a string
        return "Unknown Regime: Analyzing..."


def get_regime_color_class(internal_regime_name: str) -> str:
    """
    PYDANTIC-FIRST: Get Bootstrap color class for regime based on internal name.

    Args:
        internal_regime_name: The internal regime name from the system

    Returns:
        Bootstrap color class (success, danger, warning, info, secondary)
    """
    # PYDANTIC-FIRST: Handle None/empty values with proper defaults
    if not internal_regime_name or internal_regime_name in [None, "None", "UNKNOWN", ""]:
        return "secondary"

    try:
        regime_upper = internal_regime_name.upper()
        if "BULL" in regime_upper or "POSITIVE" in regime_upper:
            return "success"
        elif "BEAR" in regime_upper or "NEGATIVE" in regime_upper:
            return "danger"
        elif "UNCLEAR" in regime_upper or "TRANSITION" in regime_upper:
            return "warning"
        elif "VOLATILE" in regime_upper or "VOL_EXPANSION" in regime_upper:
            return "info"
        else:
            return "secondary"
    except AttributeError:
        # Handle case where internal_regime_name is not a string
        return "secondary"


def get_regime_icon(internal_regime_name: str) -> str:
    """
    PYDANTIC-FIRST: Get appropriate emoji icon for regime based on internal name.

    Args:
        internal_regime_name: The internal regime name from the system

    Returns:
        Emoji icon representing the regime
    """
    # PYDANTIC-FIRST: Handle None/empty values with proper defaults
    if not internal_regime_name or internal_regime_name in [None, "None", "UNKNOWN", ""]:
        return "❓"

    try:
        regime_upper = internal_regime_name.upper()
        if "BULL" in regime_upper:
            return "🚀"
        elif "BEAR" in regime_upper:
            return "🐻"
        elif "VOL_EXPANSION" in regime_upper or "VOLATILE" in regime_upper:
            return "🌪️"
        elif "SIDEWAYS" in regime_upper or "CONSOLIDATION" in regime_upper:
            return "⚖️"
        elif "UNCLEAR" in regime_upper or "TRANSITION" in regime_upper:
            return "❓"
        elif "VANNA" in regime_upper:
            return "⚡"
        elif "AMBUSH" in regime_upper or "FOMC" in regime_upper:
            return "🎯"
        elif "DEMAND" in regime_upper or "BUY" in regime_upper:
            return "🟢"
        elif "SELL" in regime_upper:
            return "🔴"
        else:
            return "📊"
    except AttributeError:
        # Handle case where internal_regime_name is not a string
        return "❓"


def get_regime_blurb(internal_regime_name: str) -> str:
    """
    PYDANTIC-FIRST: Return the user-facing blurb for the given regime.
    Args:
        internal_regime_name: The internal regime name from the system
    Returns:
        User-facing blurb string
    """
    # PYDANTIC-FIRST: Handle None/empty values with proper defaults
    if not internal_regime_name or internal_regime_name in [None, "None", "UNKNOWN", ""]:
        return "🧠 Market Regime Engine: Currently analyzing market conditions. Please wait for regime classification to complete."

    regime_blurb_mapping = {
        "REGIME_SPX_0DTE_FRIDAY_EOD_VANNA_CASCADE_POTENTIAL_BULLISH": "💥 Vanna Squeeze: Bullish Cascade\nSYSTEM STATE ANALYSIS: The system has entered a high-urgency 0DTE (Zero-Day-to-Expiration) state. The primary trigger is an explosive reading in vri_0dte (0DTE Volatility Regime Indicator), confirmed by a vvr_0dte (Vanna-Vomma Ratio) that shows Vanna's influence is dominant. This means dealers, who are likely short gamma and short Vanna (from selling calls), are being forced to hedge against a rapid rise in implied volatility. To hedge, they must aggressively buy the underlying stock, which pushes the stock price up, which in turn increases the options' deltas and volatility further, creating a violent, self-reinforcing upward spiral. High vci_0dte (Vanna Concentration Index) at key strikes acts as rocket fuel for this cascade.\n💡 OPERATOR TACTICAL DIRECTIVE: THIS IS A TIDAL WAVE, NOT A RIPPLE. Do NOT attempt to short or fade this move until the regime clears. The highest probability action is to trade with the cascade, targeting short, aggressive profits. Use call options or call debit spreads. Because these moves are parabolic and exhaust quickly, use aggressive profit-taking and be prepared to exit the entire position on the first sign of stalling momentum (e.g., a sharp downturn in VAPI-FA). Risk is extreme; the reversal can be as violent as the ascent. This is a predator's environment—strike fast, feed, and disappear.",
        "REGIME_SPY_PRE_FOMC_VOL_COMPRESSION_WITH_DWFD_ACCUMULATION": "🎯 Apex Ambush: Smart Money Loading\nSYSTEM STATE ANALYSIS: The system has detected a state of profound divergence. The market surface appears calm, often in a pre-catalyst environment like pre-FOMC, with low realized volatility and contracting VRI 2.0. However, beneath this quiet surface, the DWFD (Delta-Weighted Flow Divergence) is showing significant, high-conviction positioning. This means 'smart money' or institutional players are not idle; they are actively accumulating positions ('loading') in a specific direction while retail or un-informed participants are lulled into complacency. The system is flagging a potential ambush being set by apex predators.\n💡 OPERATOR TACTICAL DIRECTIVE: THE SURFACE IS A LIE; THE FLOW IS THE TRUTH. The prevailing trend is irrelevant. Your directive is to align with the direction of the DWFD signal. If DWFD shows bullish accumulation, prepare for an upside break. If bearish, prepare for a downside break. This is an opportunity to position before the catalyst with defined-risk strategies (e.g., debit spreads, credit spreads) to capitalize on the move. Be patient, but ready to strike quickly when the catalyst hits.",
        "REGIME_HIGH_VAPI_FA_BULLISH_MOMENTUM_UNIVERSAL": "⚡ Alpha Surge: Bullish Conviction\nSYSTEM STATE ANALYSIS: The system is registering a surge in VAPI-FA (Volatility Adjusted Price Imbalance - Flow Acceleration) and other bullish momentum signals. This is a classic 'momentum ignition' regime, where institutional flows and positive feedback loops drive persistent upside.\n💡 OPERATOR TACTICAL DIRECTIVE: RIDE THE WAVE, BUT WATCH FOR EXHAUSTION. This is a high-probability environment for trend-following strategies. Use trailing stops and scale out profits as momentum wanes. Avoid counter-trend trades until momentum indicators show clear reversal.",
        "REGIME_ADAPTIVE_STRUCTURE_BREAKDOWN_WITH_DWFD_CONFIRMATION_BEARISH_UNIVERSAL": "🔻 Structure Breach: Bearish Confirmed\nSYSTEM STATE ANALYSIS: The system has detected a breakdown in key structural support levels, confirmed by negative DWFD (Delta-Weighted Flow Divergence). This is a high-conviction bearish regime, often accompanied by increased volatility and rapid downside moves.\n💡 OPERATOR TACTICAL DIRECTIVE: SHORT AGGRESSIVELY, BUT MANAGE RISK. This is an environment for put options, debit spreads, or shorting. Use stop losses and be alert for short-covering rallies. Do not overstay bearish trades as volatility can reverse quickly.",
        "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH": "🔥 Ignition Point: Volatile Up\nSYSTEM STATE ANALYSIS: Volatility expansion is imminent, with bullish bias. The system is detecting a spike in VRI0DTE and related volatility metrics.\n💡 OPERATOR TACTICAL DIRECTIVE: PREPARE FOR FAST MOVES. Use options strategies that benefit from volatility expansion (e.g., straddles, strangles, long calls). Take profits quickly as volatility can mean-revert.",
        "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH": "🔥 Ignition Point: Volatile Down\nSYSTEM STATE ANALYSIS: Volatility expansion is imminent, with bearish bias. The system is detecting a spike in VRI0DTE and related volatility metrics.\n💡 OPERATOR TACTICAL DIRECTIVE: PREPARE FOR FAST MOVES. Use options strategies that benefit from volatility expansion (e.g., straddles, strangles, long puts). Take profits quickly as volatility can mean-revert.",
        "REGIME_NVP_STRONG_BUY_IMBALANCE_AT_KEY_STRIKE": "🟢 Demand Wall: Flow Support\nSYSTEM STATE ANALYSIS: Strong buy imbalance detected at a key strike, indicating institutional support.\n💡 OPERATOR TACTICAL DIRECTIVE: CONSIDER LONG POSITIONS NEAR SUPPORT. Use defined-risk strategies and monitor for breakdowns below the key strike.",
        "REGIME_EOD_HEDGING_PRESSURE_BUY": "🟢 Closing Imbalance: Systemic Buying\nSYSTEM STATE ANALYSIS: End-of-day hedging pressure is driving systemic buying.\n💡 OPERATOR TACTICAL DIRECTIVE: CONSIDER LATE-DAY LONGS, BUT EXIT BEFORE CLOSE. These moves can reverse sharply after the close.",
        "REGIME_EOD_HEDGING_PRESSURE_SELL": "🔴 Closing Imbalance: Systemic Selling\nSYSTEM STATE ANALYSIS: End-of-day hedging pressure is driving systemic selling.\n💡 OPERATOR TACTICAL DIRECTIVE: CONSIDER LATE-DAY SHORTS, BUT EXIT BEFORE CLOSE. These moves can reverse sharply after the close.",
        "REGIME_SIDEWAYS_MARKET": "⚖️ Consolidation: Tactical Chop\nSYSTEM STATE ANALYSIS: The market is range-bound with no clear directional bias.\n💡 OPERATOR TACTICAL DIRECTIVE: USE MEAN REVERSION STRATEGIES. Avoid trend trades. Take quick profits and use tight stops.",
        "REGIME_HIGH_VOLATILITY": "🌪️ Chaos State: Elevated Risk\nSYSTEM STATE ANALYSIS: The system is detecting high volatility and unpredictable price action.\n💡 OPERATOR TACTICAL DIRECTIVE: REDUCE POSITION SIZE AND USE OPTIONS FOR DEFINED RISK. Avoid overtrading and be prepared for whipsaws.",
        "REGIME_UNCLEAR_OR_TRANSITIONING": "❓ Transition State: Standby\nSYSTEM STATE ANALYSIS: The system is unable to classify the current regime with high confidence.\n💡 OPERATOR TACTICAL DIRECTIVE: STAY DEFENSIVE. Wait for clearer signals before taking new positions. Use small size or stay in cash."
    }

    try:
        return regime_blurb_mapping.get(internal_regime_name, "🧠 Market Regime Engine: Analyzes current market conditions using multiple metrics. Helps determine optimal strategy types and risk parameters. Green = Bullish conditions, Red = Bearish conditions, Yellow = Transitional/Unclear.")
    except (AttributeError, TypeError):
        # Handle case where internal_regime_name is not a string
        return "🧠 Market Regime Engine: Currently analyzing market conditions. Please wait for regime classification to complete."
