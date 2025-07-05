"""AI Hub Layout Components Module for EOTS v2.5
============================================

This module contains reusable UI components specifically for the AI Hub layout including:
- Styling constants
- Card components
- Button components
- Text styling utilities

PYDANTIC-FIRST REFACTOR: All styling functions now return Pydantic V2 models
instead of dictionaries to ensure type safety and strict compliance.

Author: EOTS v2.5 Development Team
Version: 2.5.1
"""

import logging
from typing import Dict, Any, Optional

from dash import html

# Import Pydantic models for styling - NO DICT ACCEPTANCE
from data_models import CardStyle, TypographyStyle, BadgeStyle

logger = logging.getLogger(__name__)

# ===== AI DASHBOARD STYLING CONSTANTS =====
# Exact styling from original to maintain visual consistency

AI_COLORS = {
    'primary': '#00d4ff',      # Electric Blue - Main brand color
    'secondary': '#ffd93d',    # Golden Yellow - Secondary highlights
    'accent': '#ff6b6b',       # Coral Red - Alerts and warnings
    'success': '#6bcf7f',      # Green - Positive values
    'danger': '#ff4757',       # Red - Negative values
    'warning': '#ffa726',      # Orange - Caution
    'info': '#42a5f5',         # Light Blue - Information
    'dark': '#ffffff',         # White text for dark theme
    'light': 'rgba(255, 255, 255, 0.1)',  # Light overlay for dark theme
    'muted': 'rgba(255, 255, 255, 0.6)',  # Muted white text
    'card_bg': 'rgba(255, 255, 255, 0.05)', # Dark card background
    'card_border': 'rgba(255, 255, 255, 0.1)' # Subtle border
}

AI_TYPOGRAPHY = {
    'title_size': '1.5rem',
    'subtitle_size': '1.2rem',
    'body_size': '0.9rem',
    'small_size': '0.8rem',
    'tiny_size': '0.7rem',
    'title_weight': '600',
    'subtitle_weight': '500',
    'body_weight': '400'
}

AI_SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '12px',
    'lg': '16px',
    'xl': '24px',
    'xxl': '32px'
}

AI_EFFECTS = {
    'card_shadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
    'card_shadow_hover': '0 12px 48px rgba(0, 0, 0, 0.4)',
    'box_shadow': '0 8px 32px rgba(0, 212, 255, 0.1)',
    'shadow': '0 4px 16px rgba(0, 0, 0, 0.2)',
    'shadow_lg': '0 8px 32px rgba(0, 0, 0, 0.3)',
    'border_radius': '16px',
    'border_radius_sm': '8px',
    'backdrop_blur': 'blur(20px)',
    'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    'gradient_bg': 'linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(20, 20, 20, 0.9) 50%, rgba(0, 0, 0, 0.8) 100%)',
    'glass_bg': 'rgba(0, 0, 0, 0.4)',
    'glass_border': '1px solid rgba(255, 255, 255, 0.1)'
}

# ===== CARD STYLING FUNCTIONS (PYDANTIC-FIRST) =====

def get_card_style(variant: str = 'default') -> CardStyle:
    """
    Get unified card styling as a Pydantic model.
    No dicts are returned.
    """
    base_props = {
        'borderRadius': '15px',
        'padding': '20px',
        'color': '#ffffff'
    }
    
    if variant in ('analysis', 'primary'):
        return CardStyle(
            backgroundColor='linear-gradient(145deg, #1e1e2e, #2a2a3e)',
            border='1px solid #00d4ff',
            boxShadow='0 8px 32px rgba(0, 212, 255, 0.1)',
            **base_props
        )
    elif variant in ('recommendations', 'secondary'):
        return CardStyle(
            backgroundColor='linear-gradient(145deg, #2e1e1e, #3e2a2a)',
            border='1px solid #ffd93d',
            boxShadow='0 8px 32px rgba(255, 217, 61, 0.1)',
            **base_props
        )
    elif variant in ('regime', 'success'):
        return CardStyle(
            backgroundColor='linear-gradient(145deg, #1e2e1e, #2a3e2a)',
            border='1px solid #6bcf7f',
            boxShadow='0 8px 32px rgba(107, 207, 127, 0.1)',
            **base_props
        )
    else:  # default
        return CardStyle(
            backgroundColor='linear-gradient(145deg, #1e1e1e, #2a2a2a)',
            border='1px solid rgba(255, 255, 255, 0.1)',
            boxShadow='0 8px 32px rgba(0, 0, 0, 0.3)',
            **base_props
        )

# ===== CORE UI COMPONENTS =====

def create_placeholder_card(title: str, message: str) -> html.Div:
    """Create a placeholder card for components that aren't available."""
    title_style = TypographyStyle(
        color=AI_COLORS['dark'],
        fontSize=AI_TYPOGRAPHY['title_size'],
        fontWeight=AI_TYPOGRAPHY['title_weight']
    )
    message_style = TypographyStyle(
        color=AI_COLORS['muted'],
        fontSize=AI_TYPOGRAPHY['body_size'],
        lineHeight="1.5"
    )
    
    # .model_dump() is used here at the boundary with the Dash library
    return html.Div([
        html.Div([
            html.H4(title, className="card-title mb-3", style=title_style.model_dump(exclude_none=True)),
            html.P(message, className="text-muted", style={**message_style.model_dump(exclude_none=True), "marginBottom": "0"})
        ], style=get_card_style('default').model_dump(exclude_none=True))
    ], className="ai-placeholder-card")


def create_quick_action_buttons(bundle_data, symbol: str) -> html.Div:
    """Create quick action buttons for AI dashboard."""
    try:
        button_style = {"fontSize": AI_TYPOGRAPHY['small_size']}
        return html.Div([
            html.Div([
                html.Button([
                    html.I(className="fas fa-refresh me-2"),
                    "Refresh Analysis"
                ], className="btn btn-outline-primary btn-sm me-2", style={**button_style, "borderColor": AI_COLORS['primary'], "color": AI_COLORS['primary']}),
                html.Button([
                    html.I(className="fas fa-download me-2"),
                    "Export Data"
                ], className="btn btn-outline-secondary btn-sm me-2", style={**button_style, "borderColor": AI_COLORS['secondary'], "color": AI_COLORS['secondary']}),
                html.Button([
                    html.I(className="fas fa-cog me-2"),
                    "Settings"
                ], className="btn btn-outline-info btn-sm", style={**button_style, "borderColor": AI_COLORS['info'], "color": AI_COLORS['info']})
            ], className="d-flex flex-wrap gap-2")
        ], className="quick-actions mb-3")

    except Exception as e:
        logger.error(f"Error creating quick action buttons: {str(e)}")
        return html.Div("Error creating action buttons")


def get_unified_text_style(text_type: str) -> TypographyStyle:
    """
    Get unified text styling as a Pydantic model.
    Replaces dict.get() with explicit if/elif/else for type safety.
    """
    if text_type == "title":
        return TypographyStyle(
            fontSize=AI_TYPOGRAPHY['title_size'],
            fontWeight=AI_TYPOGRAPHY['title_weight'],
            color=AI_COLORS['dark']
        )
    elif text_type == "subtitle":
        return TypographyStyle(
            fontSize=AI_TYPOGRAPHY['subtitle_size'],
            fontWeight=AI_TYPOGRAPHY['subtitle_weight'],
            color=AI_COLORS['dark']
        )
    elif text_type == "muted":
        return TypographyStyle(
            fontSize=AI_TYPOGRAPHY['small_size'],
            color=AI_COLORS['muted'],
            lineHeight="1.4"
        )
    elif text_type == "small":
        return TypographyStyle(
            fontSize=AI_TYPOGRAPHY['small_size'],
            color=AI_COLORS['dark'],
            lineHeight="1.3"
        )
    elif text_type == "danger":
        return TypographyStyle(
            fontSize=AI_TYPOGRAPHY['body_size'],
            color=AI_COLORS['danger'],
            lineHeight="1.5"
        )
    else:  # Default to "body"
        return TypographyStyle(
            fontSize=AI_TYPOGRAPHY['body_size'],
            color=AI_COLORS['dark'],
            lineHeight="1.5"
        )


def get_unified_badge_style(badge_style: str = 'success') -> BadgeStyle:
    """
    Get unified badge styling as a Pydantic model.
    Replaces dict.update() with direct model instantiation.
    """
    if badge_style == 'warning':
        bg_color, text_color = AI_COLORS['warning'], "#000000"
    elif badge_style == 'danger':
        bg_color, text_color = AI_COLORS['danger'], "#ffffff"
    elif badge_style == 'primary':
        bg_color, text_color = AI_COLORS['primary'], "#000000"
    else:  # Default to 'success'
        bg_color, text_color = AI_COLORS['success'], "#000000"

    return BadgeStyle(
        fontSize=AI_TYPOGRAPHY['tiny_size'],
        padding=f"{AI_SPACING['xs']} {AI_SPACING['sm']}",
        borderRadius=AI_EFFECTS['border_radius_sm'],
        fontWeight=AI_TYPOGRAPHY['subtitle_weight'],
        backgroundColor=bg_color,
        color=text_color
    )


def create_clickable_title_with_info(title: str, info_id: str, info_content: str,
                                   title_style_model: Optional[TypographyStyle] = None,
                                   badge_text: Optional[str] = None,
                                   badge_style: str = 'success') -> html.Details:
    """
    Create a clickable title using Pydantic models for styling.
    Accepts a TypographyStyle model instead of a dict.
    """
    # Default title style as a Pydantic model
    default_title_style = TypographyStyle(
        color=AI_COLORS['dark'],
        fontSize=AI_TYPOGRAPHY['title_size'],
        fontWeight=AI_TYPOGRAPHY['title_weight']
    )
    
    final_title_style = title_style_model if title_style_model else default_title_style

    # Create the summary content (clickable title)
    summary_content = [html.Span(title, style=final_title_style.model_dump(exclude_none=True))]
    if badge_text is not None:
        badge_model = get_unified_badge_style(badge_style)
        summary_content.extend([
            html.Span(" "),
            html.Span(badge_text, className="badge", style=badge_model.model_dump(exclude_none=True))
        ])

    # Define styles for summary and content using Pydantic models where possible
    summary_title_style = {
        **final_title_style.model_dump(exclude_none=True),
        "margin": "0",
        "cursor": "pointer",
        "userSelect": "none",
        "transition": AI_EFFECTS['transition']
    }
    
    content_text_style = TypographyStyle(
        fontSize=AI_TYPOGRAPHY['small_size'],
        lineHeight="1.6",
        color=AI_COLORS['dark']
    )

    content_container_style = {
        "margin": "0",
        "padding": f"{AI_SPACING['md']} {AI_SPACING['lg']}",
        "background": "rgba(255, 255, 255, 0.05)",
        "borderRadius": AI_EFFECTS['border_radius_sm'],
        "border": "1px solid rgba(255, 255, 255, 0.1)",
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    }

    return html.Details([
        # Summary (clickable title)
        html.Summary([
            html.H5(summary_content, className="mb-0", style=summary_title_style)
        ], style={"cursor": "pointer", "listStyle": "none", "outline": "none"}),

        # Collapsible content
        html.Div([
            html.P(
                info_content,
                style={**content_text_style.model_dump(exclude_none=True), **content_container_style}
            )
        ], style={"marginTop": AI_SPACING['sm'], "animation": "fadeIn 0.3s ease-in-out"})
    ], id=info_id, style={"marginBottom": AI_SPACING['md']})
