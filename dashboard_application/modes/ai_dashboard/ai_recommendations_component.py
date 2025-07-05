# dashboard_application/modes/ai_dashboard/ai_recommendations_component.py
"""
AI Trade Recommendations Component for EOTS AI Hub v2.5
========================================================

PYDANTIC-FIRST & ZERO DICT ACCEPTANCE.

This module is dedicated to generating and displaying AI-driven trade recommendations.
As a high-priority component, it provides actionable insights derived from the synthesis
of all HuiHui expert analyses.

Key Features:
1.  **Pydantic-First Transformation**: Converts raw `MOEUnifiedResponseV2_5` into a
    strict `AIRecommendationsPanelModel`.
2.  **Sophisticated Generation Logic**: A dedicated `RecommendationGenerator` class
    synthesizes expert data to create bullish, bearish, and neutral trade ideas.
3.  **Actionable & Rich UI**: Each recommendation is displayed in a styled card
    with confidence, multi-point rationale, and risk management guidance.
4.  **Strict Compliance**: Adheres to the ZERO DICT ACCEPTANCE policy, ensuring
    end-to-end type safety and data integrity.

Author: EOTS v2.5 Development Team
Version: 2.5.1
"""

import logging
from datetime import datetime
import uuid
from typing import Optional, List, Tuple

from dash import html, dcc

# EOTS Pydantic Models: The single source of truth for all data structures
from data_models import (
    MOEUnifiedResponseV2_5,
    AIRecommendationsPanelModel,
    AIRecommendationItem,
    PanelConfigModel,
    ComponentStatus,
    PanelType,
    RecommendationStrength,
    MarketRegimeAnalysisDetails,
    OptionsFlowAnalysisDetails,
    SentimentAnalysisDetails,
    TypographyStyle
)

# Import shared components and styling constants
from .components import (
    create_placeholder_card,
    get_card_style,
    get_unified_text_style,
    get_unified_badge_style,
    AI_COLORS
)

logger = logging.getLogger(__name__)


# --- 1. Core Data Transformation Logic ---

def create_recommendations_model_from_analysis(
    analysis_bundle: Optional[MOEUnifiedResponseV2_5],
    symbol: str,
    underlying_price: float
) -> AIRecommendationsPanelModel:
    """
    Transforms the raw MOEUnifiedResponseV2_5 from the orchestrator into a
    strictly validated AIRecommendationsPanelModel. This is the primary data
    processing function for the recommendations panel.

    Args:
        analysis_bundle: The unified response from the HuiHui experts.
        symbol: The target trading symbol.
        underlying_price: The current price of the underlying asset.

    Returns:
        A fully populated and validated AIRecommendationsPanelModel.
    """
    panel_config = PanelConfigModel(
        id=f"ai-recommendations-{symbol}",
        title="AI-Driven Trade Recommendations",
        panel_type=PanelType.TRADE_RECOMMENDATIONS
    )

    if not analysis_bundle or not analysis_bundle.expert_responses or not underlying_price:
        logger.warning(f"Recommendations: Insufficient data for {symbol}. Returning loading state.")
        return AIRecommendationsPanelModel(panel_config=panel_config)

    try:
        # Extract and validate details from the three core experts
        regime_details = _get_expert_details(analysis_bundle, "market_regime", MarketRegimeAnalysisDetails)
        flow_details = _get_expert_details(analysis_bundle, "options_flow", OptionsFlowAnalysisDetails)
        sentiment_details = _get_expert_details(analysis_bundle, "sentiment", SentimentAnalysisDetails)

        # Instantiate the generator with all necessary data
        generator = RecommendationGenerator(
            regime=regime_details,
            flow=flow_details,
            sentiment=sentiment_details,
            symbol=symbol,
            price=underlying_price
        )

        # Generate the list of recommendation items
        recommendations = generator.generate()

        panel_config.status = ComponentStatus.OK
        panel_config.last_updated = datetime.now()

        return AIRecommendationsPanelModel(
            panel_config=panel_config,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Recommendations: Failed to create model for {symbol}. Error: {e}", exc_info=True)
        panel_config.status = ComponentStatus.ERROR
        return AIRecommendationsPanelModel(
            panel_config=panel_config,
            recommendations=[]
        )


# --- 2. Sophisticated Recommendation Generation Logic ---

class RecommendationGenerator:
    """
    Encapsulates the logic for generating trade recommendations by synthesizing
    analysis from all experts.
    """
    def __init__(self, regime, flow, sentiment, symbol, price):
        self.regime = regime
        self.flow = flow
        self.sentiment = sentiment
        self.symbol = symbol
        self.price = price

    def generate(self) -> List[AIRecommendationItem]:
        """
        Main generation method. Identifies the most probable scenarios and
        creates corresponding recommendation items.
        """
        if not all([self.regime, self.flow, self.sentiment]):
            return []

        # --- High-Conviction Bullish Scenario ---
        if (self.regime.trend_direction == "BULLISH" and
            self.regime.vri_score > 60 and
            self.flow.institutional_probability > 0.7 and
            self.sentiment.overall_sentiment_score > 0.5):
            return [self._generate_high_conviction_bullish()]

        # --- High-Conviction Bearish Scenario ---
        if (self.regime.trend_direction == "BEARISH" and
            self.regime.vri_score > 60 and
            self.flow.institutional_probability > 0.7 and
            self.sentiment.overall_sentiment_score < -0.5):
            return [self._generate_high_conviction_bearish()]
        
        # --- Volatility Expansion Scenario ---
        if self.regime.volatility_level in ["HIGH", "EXTREME"] and self.regime.trend_direction == "NEUTRAL":
             return [self._generate_volatility_play()]

        # --- Default Neutral/Observational Scenario ---
        return [self._generate_neutral_observation()]

    def _generate_high_conviction_bullish(self) -> AIRecommendationItem:
        confidence = (self.regime.vri_score/100 + self.flow.institutional_probability + self.sentiment.overall_sentiment_score) / 3
        return AIRecommendationItem(
            id=str(uuid.uuid4()),
            strategy_name="Bullish Trend Continuation",
            direction="Bullish",
            instrument_type="Call",
            strength=RecommendationStrength.HIGH,
            confidence_score=confidence,
            rationale=[
                f"Strong Bullish Market Regime confirmed (VRI: {self.regime.vri_score:.0f}).",
                f"High institutional probability ({self.flow.institutional_probability:.0%}) in options flow.",
                f"Positive market sentiment (Score: {self.sentiment.overall_sentiment_score:.2f})."
            ],
            target_price=self.price * 1.05, # 5% target
            stop_loss=self.price * 0.98, # 2% stop-loss
            timeframe="1-3 Days"
        )

    def _generate_high_conviction_bearish(self) -> AIRecommendationItem:
        confidence = (self.regime.vri_score/100 + self.flow.institutional_probability + abs(self.sentiment.overall_sentiment_score)) / 3
        return AIRecommendationItem(
            id=str(uuid.uuid4()),
            strategy_name="Bearish Trend Continuation",
            direction="Bearish",
            instrument_type="Put",
            strength=RecommendationStrength.HIGH,
            confidence_score=confidence,
            rationale=[
                f"Strong Bearish Market Regime confirmed (VRI: {self.regime.vri_score:.0f}).",
                f"High institutional probability ({self.flow.institutional_probability:.0%}) driving downside.",
                f"Negative market sentiment (Score: {self.sentiment.overall_sentiment_score:.2f})."
            ],
            target_price=self.price * 0.95, # 5% target
            stop_loss=self.price * 1.02, # 2% stop-loss
            timeframe="1-3 Days"
        )
    
    def _generate_volatility_play(self) -> AIRecommendationItem:
        confidence = self.regime.transition_probability
        return AIRecommendationItem(
            id=str(uuid.uuid4()),
            strategy_name="Volatility Expansion",
            direction="Neutral",
            instrument_type="Spread",
            strength=RecommendationStrength.MEDIUM,
            confidence_score=confidence,
            rationale=[
                f"Market regime indicates {self.regime.volatility_level} volatility.",
                f"Trend direction is neutral, suggesting price chop or breakout potential.",
                "Consider straddles or strangles to play non-directional movement."
            ],
            timeframe="1-5 Days"
        )

    def _generate_neutral_observation(self) -> AIRecommendationItem:
        return AIRecommendationItem(
            id=str(uuid.uuid4()),
            strategy_name="Observe Market",
            direction="Neutral",
            instrument_type="Underlying",
            strength=RecommendationStrength.LOW,
            confidence_score=0.4,
            rationale=[
                "Expert analyses show conflicting or weak signals.",
                f"Market regime ({self.regime.regime_name}) does not indicate a strong directional bias.",
                "Recommend waiting for a clearer confluence of signals before committing capital."
            ],
            timeframe="Intraday"
        )


# --- 3. UI Rendering Logic ---

def create_ai_recommendations_panel(
    recommendations_model: Optional[AIRecommendationsPanelModel]
) -> html.Div:
    """
    Assembles the complete AI Recommendations panel component for the AI Hub layout.

    Args:
        recommendations_model: A validated AIRecommendationsPanelModel.

    Returns:
        A Dash html.Div containing the recommendation cards or a placeholder.
    """
    if not recommendations_model or recommendations_model.panel_config.status in [ComponentStatus.LOADING, ComponentStatus.UNKNOWN]:
        return create_placeholder_card("AI Recommendations", "Awaiting analysis data...")
    
    if recommendations_model.panel_config.status == ComponentStatus.ERROR:
        return create_placeholder_card("AI Recommendations", "An error occurred while generating recommendations.")

    if not recommendations_model.recommendations:
        return create_placeholder_card("AI Recommendations", "No specific trade recommendations at this time. Monitor for changes.")

    # Create a card for each recommendation
    cards = [
        _create_recommendation_card(item) for item in recommendations_model.recommendations
    ]
    
    return html.Div([
        html.H4(
            recommendations_model.panel_config.title,
            className="card-title mb-3",
            style=get_unified_text_style("title").model_dump(exclude_none=True)
        ),
        html.Div(cards, className="row")
    ])

def _create_recommendation_card(item: AIRecommendationItem) -> html.Div:
    """Renders a single AIRecommendationItem into a styled Dash card."""
    
    # Determine card styling based on direction
    if item.direction == "Bullish":
        card_variant = 'success'
        icon_class = "fas fa-arrow-up text-success"
    elif item.direction == "Bearish":
        card_variant = 'danger'
        icon_class = "fas fa-arrow-down text-danger"
    else:
        card_variant = 'secondary'
        icon_class = "fas fa-circle-notch text-warning"

    card_style = get_card_style(card_variant)
    # Use a less intense background for the card body
    card_style.backgroundColor = 'rgba(30, 30, 40, 0.8)'

    return html.Div(
        className="col-md-6 col-lg-4 mb-4",
        children=html.Div(
            className="card h-100",
            style=card_style.model_dump(exclude_none=True),
            children=[
                html.Div(className="card-body", children=[
                    # --- Card Header ---
                    html.Div(className="d-flex justify-content-between align-items-center mb-2", children=[
                        html.H5(item.strategy_name, className="card-title mb-0", style={"color": AI_COLORS['dark']}),
                        html.I(className=icon_class, style={"fontSize": "1.5rem"})
                    ]),
                    html.Div(className="d-flex justify-content-between mb-3", children=[
                        html.Span(
                            item.strength.value,
                            style=get_unified_badge_style(card_variant).model_dump(exclude_none=True)
                        ),
                        html.Span(f"Confidence: {item.confidence_score:.0%}", style={"fontWeight": "bold", "color": AI_COLORS['primary']})
                    ]),
                    
                    # --- Rationale ---
                    html.H6("Rationale:", className="mt-4", style={"color": AI_COLORS['secondary']}),
                    html.Ul(
                        [html.Li(point, style={"fontSize": "0.85rem"}) for point in item.rationale],
                        className="list-unstyled ps-3"
                    ),

                    # --- Actionable Targets ---
                    html.H6("Actionable Targets:", className="mt-4", style={"color": AI_COLORS['secondary']}),
                    _create_target_row("Instrument", item.instrument_type),
                    _create_target_row("Timeframe", item.timeframe),
                    _create_target_row("Target Price", f"${item.target_price:.2f}" if item.target_price else "N/A"),
                    _create_target_row("Stop Loss", f"${item.stop_loss:.2f}" if item.stop_loss else "N/A"),

                    # --- Risk Profile ---
                    html.H6("Risk Profile:", className="mt-4", style={"color": AI_COLORS['secondary']}),
                    _create_target_row("Guidance", "Use small position size. Confirm with price action.")
                ])
            ]
        )
    )

# --- Helper Functions ---

def _get_expert_details(bundle, expert_id, model_class):
    """Safely extracts and validates expert details from the bundle."""
    for resp in bundle.expert_responses:
        if resp.expert_id == expert_id and resp.success and resp.response_data.details:
            try:
                # The details can be a dict, so we must validate it into our Pydantic model
                return model_class.model_validate(resp.response_data.details)
            except Exception as e:
                logger.warning(f"Could not validate details for expert {expert_id}: {e}")
    return None

def _create_target_row(label: str, value: str) -> html.Div:
    """Creates a styled row for displaying a label and value."""
    return html.Div(
        className="d-flex justify-content-between",
        style={"padding": "4px 0", "borderBottom": f"1px solid {AI_COLORS['card_border']}"},
        children=[
            html.Span(f"{label}:", style={"fontWeight": "500", "color": AI_COLORS['muted']}),
            html.Span(value, style={"fontWeight": "bold", "color": AI_COLORS['dark']})
        ]
    )
