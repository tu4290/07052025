"""
Historical Data Storage Manager for EOTS v2.5
==============================================

This module provides functions to store historical data for long-term analysis
and pattern recognition using the existing DatabaseManagerV2_5 connection.

Author: EOTS v2.5 Development Team
"""

import logging
import json
from datetime import date
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HistoricalDataStorageV2_5:
    """Manages historical data storage for long-term analysis."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logger.getChild(self.__class__.__name__)
    
    def store_daily_regime_snapshot(self, symbol: str, regime_data: Dict[str, Any]) -> bool:
        """Store daily regime snapshot for historical analysis."""
        try:
            # Prepare data
            snapshot_data = {
                'symbol': symbol,
                'date': regime_data.get('date', date.today()),
                'primary_regime': regime_data.get('primary_regime', 'REGIME_UNCLEAR_OR_TRANSITIONING'),
                'regime_confidence_score': min(max(regime_data.get('confidence_score', 0.0), 0.0), 1.0),
                'regime_duration_minutes': regime_data.get('duration_minutes', 0),
                'regime_transition_count': regime_data.get('transition_count', 0),
                'secondary_regimes': json.dumps(regime_data.get('secondary_regimes', [])),
                'market_conditions': json.dumps(regime_data.get('market_conditions', {})),
                'volatility_environment': regime_data.get('volatility_environment', 'NORMAL'),
                'flow_intensity_score': min(max(regime_data.get('flow_intensity', 0.0), 0.0), 1.0)
            }
            
            # Insert or update
            sql = """
            INSERT INTO daily_regime_snapshots 
            (symbol, date, primary_regime, regime_confidence_score, regime_duration_minutes,
             regime_transition_count, secondary_regimes, market_conditions, 
             volatility_environment, flow_intensity_score)
            VALUES (%(symbol)s, %(date)s, %(primary_regime)s, %(regime_confidence_score)s,
                    %(regime_duration_minutes)s, %(regime_transition_count)s, %(secondary_regimes)s,
                    %(market_conditions)s, %(volatility_environment)s, %(flow_intensity_score)s)
            ON CONFLICT (symbol, date) 
            DO UPDATE SET
                primary_regime = EXCLUDED.primary_regime,
                regime_confidence_score = EXCLUDED.regime_confidence_score,
                regime_duration_minutes = EXCLUDED.regime_duration_minutes,
                regime_transition_count = EXCLUDED.regime_transition_count,
                secondary_regimes = EXCLUDED.secondary_regimes,
                market_conditions = EXCLUDED.market_conditions,
                volatility_environment = EXCLUDED.volatility_environment,
                flow_intensity_score = EXCLUDED.flow_intensity_score
            """
            
            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, snapshot_data)
            self.logger.info(f"Stored regime snapshot for {symbol} on {snapshot_data['date']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store regime snapshot: {e}")
            return False
    
    def store_key_level_performance(self, symbol: str, level_data: Dict[str, Any]) -> bool:
        """Store key level performance data."""
        try:
            # Prepare data
            level_performance = {
                'symbol': symbol,
                'date': level_data.get('date', date.today()),
                'level_price': level_data.get('price', 0.0),
                'level_type': level_data.get('type', 'support'),
                'level_source': level_data.get('source', 'unknown'),
                'hit_count': level_data.get('hit_count', 0),
                'bounce_count': level_data.get('bounce_count', 0),
                'break_count': level_data.get('break_count', 0),
                'bounce_accuracy': min(max(level_data.get('bounce_accuracy', 0.0), 0.0), 1.0),
                'break_significance': min(max(level_data.get('break_significance', 0.0), 0.0), 1.0),
                'time_to_test_minutes': level_data.get('time_to_test', None),
                'hold_duration_minutes': level_data.get('hold_duration', None),
                'max_distance_from_level': level_data.get('max_distance', None),
                'conviction_score': min(max(level_data.get('conviction', 0.0), 0.0), 1.0),
                'market_regime_context': level_data.get('regime_context', 'UNKNOWN')
            }
            
            sql = """
            INSERT INTO key_level_performance 
            (symbol, date, level_price, level_type, level_source, hit_count, bounce_count,
             break_count, bounce_accuracy, break_significance, time_to_test_minutes,
             hold_duration_minutes, max_distance_from_level, conviction_score, market_regime_context)
            VALUES (%(symbol)s, %(date)s, %(level_price)s, %(level_type)s, %(level_source)s,
                    %(hit_count)s, %(bounce_count)s, %(break_count)s, %(bounce_accuracy)s,
                    %(break_significance)s, %(time_to_test_minutes)s, %(hold_duration_minutes)s,
                    %(max_distance_from_level)s, %(conviction_score)s, %(market_regime_context)s)
            """
            
            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, level_performance)
            self.logger.info(f"Stored key level performance for {symbol}: {level_data['type']} at {level_data['price']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store key level performance: {e}")
            return False
    
    def store_flow_pattern(self, symbol: str, pattern_data: Dict[str, Any]) -> bool:
        """Store flow pattern for pattern library."""
        try:
            # Prepare data
            pattern_info = {
                'symbol': symbol,
                'date': pattern_data.get('date', date.today()),
                'pattern_name': pattern_data.get('name', 'unknown_pattern'),
                'pattern_signature': json.dumps(pattern_data.get('signature', {})),
                'pattern_strength': min(max(pattern_data.get('strength', 0.0), 0.0), 1.0),
                'market_outcome': pattern_data.get('outcome', None),
                'follow_through_success': pattern_data.get('follow_through_success', None),
                'follow_through_magnitude': pattern_data.get('follow_through_magnitude', None),
                'regime_context': pattern_data.get('regime_context', 'UNKNOWN'),
                'volatility_environment': pattern_data.get('volatility_environment', 'NORMAL'),
                'time_horizon_minutes': pattern_data.get('time_horizon', None),
                'success_rate': min(max(pattern_data.get('success_rate', 0.0), 0.0), 1.0),
                'sample_size': pattern_data.get('sample_size', 1),
                'confidence_score': min(max(pattern_data.get('confidence', 0.0), 0.0), 1.0)
            }
            
            sql = """
            INSERT INTO flow_pattern_library 
            (symbol, date, pattern_name, pattern_signature, pattern_strength, market_outcome,
             follow_through_success, follow_through_magnitude, regime_context, volatility_environment,
             time_horizon_minutes, success_rate, sample_size, confidence_score)
            VALUES (%(symbol)s, %(date)s, %(pattern_name)s, %(pattern_signature)s, %(pattern_strength)s,
                    %(market_outcome)s, %(follow_through_success)s, %(follow_through_magnitude)s,
                    %(regime_context)s, %(volatility_environment)s, %(time_horizon_minutes)s,
                    %(success_rate)s, %(sample_size)s, %(confidence_score)s)
            """
            
            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, pattern_info)
            self.logger.info(f"Stored flow pattern for {symbol}: {pattern_data['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store flow pattern: {e}")
            return False
    
    def store_ai_performance_summary(self, symbol: str, performance_data: Dict[str, Any]) -> bool:
        """Store daily AI performance summary."""
        try:
            # Prepare data
            summary_data = {
                'date': performance_data.get('date', date.today()),
                'symbol': symbol,
                'total_predictions': performance_data.get('total_predictions', 0),
                'correct_predictions': performance_data.get('correct_predictions', 0),
                'accuracy_rate': min(max(performance_data.get('accuracy_rate', 0.0), 0.0), 1.0),
                'avg_confidence_score': min(max(performance_data.get('avg_confidence', 0.0), 0.0), 1.0),
                'regime_prediction_accuracy': json.dumps(performance_data.get('regime_accuracy', {})),
                'prediction_type_performance': json.dumps(performance_data.get('type_performance', {})),
                'high_confidence_accuracy': min(max(performance_data.get('high_conf_accuracy', 0.0), 0.0), 1.0),
                'medium_confidence_accuracy': min(max(performance_data.get('med_conf_accuracy', 0.0), 0.0), 1.0),
                'low_confidence_accuracy': min(max(performance_data.get('low_conf_accuracy', 0.0), 0.0), 1.0)
            }
            
            sql = """
            INSERT INTO ai_performance_daily_summary 
            (date, symbol, total_predictions, correct_predictions, accuracy_rate, avg_confidence_score,
             regime_prediction_accuracy, prediction_type_performance, high_confidence_accuracy,
             medium_confidence_accuracy, low_confidence_accuracy)
            VALUES (%(date)s, %(symbol)s, %(total_predictions)s, %(correct_predictions)s, %(accuracy_rate)s,
                    %(avg_confidence_score)s, %(regime_prediction_accuracy)s, %(prediction_type_performance)s,
                    %(high_confidence_accuracy)s, %(medium_confidence_accuracy)s, %(low_confidence_accuracy)s)
            ON CONFLICT (date, symbol)
            DO UPDATE SET
                total_predictions = EXCLUDED.total_predictions,
                correct_predictions = EXCLUDED.correct_predictions,
                accuracy_rate = EXCLUDED.accuracy_rate,
                avg_confidence_score = EXCLUDED.avg_confidence_score,
                regime_prediction_accuracy = EXCLUDED.regime_prediction_accuracy,
                prediction_type_performance = EXCLUDED.prediction_type_performance,
                high_confidence_accuracy = EXCLUDED.high_confidence_accuracy,
                medium_confidence_accuracy = EXCLUDED.medium_confidence_accuracy,
                low_confidence_accuracy = EXCLUDED.low_confidence_accuracy
            """
            
            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, summary_data)
            self.logger.info(f"Stored AI performance summary for {symbol} on {summary_data['date']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store AI performance summary: {e}")
            return False

    # ===== QUERY FUNCTIONS FOR HISTORICAL ANALYSIS =====

    def get_regime_pattern_analysis(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get regime pattern analysis for the last N days."""
        try:
            sql = """
            SELECT
                primary_regime,
                COUNT(*) as occurrence_count,
                AVG(regime_confidence_score) as avg_confidence,
                AVG(regime_duration_minutes) as avg_duration,
                AVG(regime_transition_count) as avg_transitions,
                AVG(flow_intensity_score) as avg_flow_intensity
            FROM daily_regime_snapshots
            WHERE symbol = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY primary_regime
            ORDER BY occurrence_count DESC
            """

            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, (symbol, days))
            results = cursor.fetchall()

            return {
                'regime_patterns': [dict(row) for row in results],
                'analysis_period_days': days,
                'total_days_analyzed': sum(row['occurrence_count'] for row in results)
            }

        except Exception as e:
            self.logger.error(f"Failed to get regime pattern analysis: {e}")
            return {}

    def get_key_level_effectiveness(self, symbol: str, level_type: str = None, days: int = 30) -> Dict[str, Any]:
        """Get key level effectiveness analysis."""
        try:
            where_clause = "WHERE symbol = %s AND date >= CURRENT_DATE - INTERVAL '%s days'"
            params = [symbol, days]

            if level_type:
                where_clause += " AND level_type = %s"
                params.append(level_type)

            sql = f"""
            SELECT
                level_type,
                COUNT(*) as total_levels,
                AVG(bounce_accuracy) as avg_bounce_accuracy,
                AVG(break_significance) as avg_break_significance,
                AVG(conviction_score) as avg_conviction,
                AVG(time_to_test_minutes) as avg_time_to_test,
                SUM(hit_count) as total_hits,
                SUM(bounce_count) as total_bounces,
                SUM(break_count) as total_breaks
            FROM key_level_performance
            {where_clause}
            GROUP BY level_type
            ORDER BY avg_bounce_accuracy DESC
            """

            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, params)
            results = cursor.fetchall()

            return {
                'level_effectiveness': [dict(row) for row in results],
                'analysis_period_days': days,
                'level_type_filter': level_type
            }

        except Exception as e:
            self.logger.error(f"Failed to get key level effectiveness: {e}")
            return {}

    def get_flow_pattern_success_rates(self, symbol: str, pattern_name: str = None, days: int = 30) -> Dict[str, Any]:
        """Get flow pattern success rates and outcomes."""
        try:
            where_clause = "WHERE symbol = %s AND date >= CURRENT_DATE - INTERVAL '%s days'"
            params = [symbol, days]

            if pattern_name:
                where_clause += " AND pattern_name = %s"
                params.append(pattern_name)

            sql = f"""
            SELECT
                pattern_name,
                COUNT(*) as pattern_count,
                AVG(success_rate) as avg_success_rate,
                AVG(pattern_strength) as avg_strength,
                AVG(confidence_score) as avg_confidence,
                COUNT(CASE WHEN follow_through_success = true THEN 1 END) as successful_follow_through,
                COUNT(CASE WHEN follow_through_success = false THEN 1 END) as failed_follow_through,
                AVG(follow_through_magnitude) as avg_magnitude
            FROM flow_pattern_library
            {where_clause}
            GROUP BY pattern_name
            ORDER BY avg_success_rate DESC
            """

            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, params)
            results = cursor.fetchall()

            return {
                'pattern_analysis': [dict(row) for row in results],
                'analysis_period_days': days,
                'pattern_filter': pattern_name
            }

        except Exception as e:
            self.logger.error(f"Failed to get flow pattern success rates: {e}")
            return {}

    def get_ai_performance_trends(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get AI performance trends over time."""
        try:
            sql = """
            SELECT
                date,
                accuracy_rate,
                avg_confidence_score,
                total_predictions,
                correct_predictions,
                high_confidence_accuracy,
                medium_confidence_accuracy,
                low_confidence_accuracy
            FROM ai_performance_daily_summary
            WHERE symbol = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
            """

            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, (symbol, days))
            results = cursor.fetchall()

            # Calculate trends
            daily_data = [dict(row) for row in results]
            if len(daily_data) >= 2:
                recent_accuracy = sum(row['accuracy_rate'] for row in daily_data[:7]) / min(7, len(daily_data))
                older_accuracy = sum(row['accuracy_rate'] for row in daily_data[7:14]) / max(1, min(7, len(daily_data) - 7))
                accuracy_trend = recent_accuracy - older_accuracy
            else:
                accuracy_trend = 0.0

            return {
                'daily_performance': daily_data,
                'accuracy_trend': accuracy_trend,
                'analysis_period_days': days,
                'total_predictions': sum(row['total_predictions'] for row in daily_data),
                'overall_accuracy': sum(row['correct_predictions'] for row in daily_data) / max(1, sum(row['total_predictions'] for row in daily_data))
            }

        except Exception as e:
            self.logger.error(f"Failed to get AI performance trends: {e}")
            return {}
