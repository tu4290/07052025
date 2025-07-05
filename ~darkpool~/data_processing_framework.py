#!/usr/bin/env python3
"""
Data Processing Framework for Darkpool Analysis using ConvexValue Metrics
Focus: SPY options for the past 5 days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class DarkpoolAnalyzer:
    """
    A framework for analyzing darkpool activity using ConvexValue metrics
    """
    
    def __init__(self, symbol="SPY", days_back=5):
        """
        Initialize the analyzer with target symbol and timeframe
        
        Parameters:
        -----------
        symbol : str
            The target symbol (default: SPY)
        days_back : int
            Number of days to analyze (default: 5)
        """
        self.symbol = symbol
        self.days_back = days_back
        self.data = None
        self.darkpool_levels = []
        self.top_levels = []
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        print(f"Initialized DarkpoolAnalyzer for {symbol}, analyzing past {days_back} days")
    
    def load_sample_data(self):
        """
        Load sample data for demonstration purposes
        In a real implementation, this would fetch actual ConvexValue data
        """
        # Generate dates for the past 5 days (excluding weekends for simplicity)
        end_date = datetime.now()
        dates = []
        days_added = 0
        while days_added < self.days_back:
            if end_date.weekday() < 5:  # Monday to Friday
                dates.append(end_date.strftime('%Y-%m-%d'))
                days_added += 1
            end_date -= timedelta(days=1)
        dates.reverse()  # Chronological order
        
        # Generate price levels around current SPY price (approximate)
        base_price = 500  # Approximate SPY price
        strikes = np.arange(base_price - 50, base_price + 50, 5)
        
        # Create sample data with key metrics
        data_rows = []
        
        for date in dates:
            for strike in strikes:
                # Generate realistic but random values for key metrics
                gxoi = np.random.gamma(10, 100000) if np.random.random() < 0.3 else np.random.gamma(2, 10000)
                dxoi = np.random.normal(0, 500000)
                volmbs_15m = np.random.normal(0, 200)
                charmxoi = np.random.normal(0, 50000)
                vannaxoi = np.random.normal(0, 30000)
                vommaxoi = np.random.normal(0, 20000)
                gxvolm = np.random.gamma(2, 1000)
                value_bs = np.random.normal(0, 100000)
                
                # Add some darkpool "hotspots" for demonstration
                if strike % 25 == 0:  # Major strikes often have more activity
                    gxoi *= 3
                    gxvolm *= 2
                
                # Add a row to our dataset
                data_rows.append({
                    'date': date,
                    'strike': strike,
                    'gxoi': gxoi,
                    'dxoi': dxoi,
                    'volmbs_15m': volmbs_15m,
                    'charmxoi': charmxoi,
                    'vannaxoi': vannaxoi,
                    'vommaxoi': vommaxoi,
                    'gxvolm': gxvolm,
                    'value_bs': value_bs,
                    'flownet': value_bs * 1.2,  # Simplified relationship
                    'vflowratio': 1 + (value_bs / 100000),  # Simplified relationship
                    'put_call_ratio': 1 + (dxoi / 1000000),  # Simplified relationship
                    'volmbs_5m': volmbs_15m * 0.4,  # Simplified relationship
                    'volmbs_30m': volmbs_15m * 1.8,  # Simplified relationship
                    'volmbs_60m': volmbs_15m * 3.2,  # Simplified relationship
                    'valuebs_15m': volmbs_15m * 100,  # Simplified relationship
                })
        
        # Convert to DataFrame
        self.data = pd.DataFrame(data_rows)
        print(f"Loaded sample data with {len(self.data)} rows")
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the data for analysis
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Add derived metrics that might be useful for darkpool analysis
        self.data['abs_gxoi'] = self.data['gxoi'].abs()
        self.data['abs_dxoi'] = self.data['dxoi'].abs()
        self.data['gxoi_to_dxoi_ratio'] = self.data['gxoi'] / self.data['abs_dxoi'].replace(0, 1)
        self.data['volmbs_ratio'] = self.data['volmbs_15m'] / self.data['volmbs_60m'].replace(0, 1)
        
        # Normalize key metrics for comparison
        for metric in ['gxoi', 'dxoi', 'volmbs_15m', 'volmbs_60m', 'charmxoi', 'vannaxoi', 'vommaxoi', 'gxvolm', 'value_bs']:
            mean = self.data[metric].mean()
            std = self.data[metric].std()
            self.data[f'{metric}_zscore'] = (self.data[metric] - mean) / std
        
        print("Data preprocessing complete")
        return self.data
    
    def identify_darkpool_levels(self):
        """
        Identify potential darkpool levels using multiple methodologies
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return []
        
        # Method 1: High Gamma Imbalance Method
        # Identifies strikes with unusually high gamma concentration
        gamma_threshold = self.data['gxoi_zscore'].quantile(0.9)
        gamma_levels = self.data[self.data['gxoi_zscore'] > gamma_threshold].copy()
        gamma_levels['method'] = 'High Gamma Imbalance'
        gamma_levels['score'] = gamma_levels['gxoi_zscore']
        
        # Method 2: Delta-Gamma Divergence Method
        # Identifies strikes where delta and gamma imbalances diverge significantly
        self.data['delta_gamma_divergence'] = (self.data['gxoi_zscore'] - self.data['dxoi_zscore']).abs()
        divergence_threshold = self.data['delta_gamma_divergence'].quantile(0.9)
        divergence_levels = self.data[self.data['delta_gamma_divergence'] > divergence_threshold].copy()
        divergence_levels['method'] = 'Delta-Gamma Divergence'
        divergence_levels['score'] = divergence_levels['delta_gamma_divergence']
        
        # Method 3: Flow Anomaly Method
        # Identifies strikes with unusual flow patterns across timeframes
        self.data['flow_anomaly'] = (
            self.data['volmbs_15m_zscore'].abs() + 
            (self.data['volmbs_15m_zscore'] - self.data['volmbs_60m_zscore']).abs()
        )
        flow_threshold = self.data['flow_anomaly'].quantile(0.9)
        flow_levels = self.data[self.data['flow_anomaly'] > flow_threshold].copy()
        flow_levels['method'] = 'Flow Anomaly'
        flow_levels['score'] = flow_levels['flow_anomaly']
        
        # Method 4: Volatility Sensitivity Method
        # Identifies strikes with high vanna and vomma exposure
        self.data['vol_sensitivity'] = self.data['vannaxoi_zscore'].abs() + self.data['vommaxoi_zscore'].abs()
        vol_threshold = self.data['vol_sensitivity'].quantile(0.9)
        vol_levels = self.data[self.data['vol_sensitivity'] > vol_threshold].copy()
        vol_levels['method'] = 'Volatility Sensitivity'
        vol_levels['score'] = vol_levels['vol_sensitivity']
        
        # Method 5: Charm-Adjusted Gamma Method
        # Identifies strikes with high gamma that are also sensitive to time decay
        self.data['charm_adjusted_gamma'] = self.data['gxoi_zscore'] * (1 + self.data['charmxoi_zscore'].abs())
        charm_threshold = self.data['charm_adjusted_gamma'].quantile(0.9)
        charm_levels = self.data[self.data['charm_adjusted_gamma'] > charm_threshold].copy()
        charm_levels['method'] = 'Charm-Adjusted Gamma'
        charm_levels['score'] = charm_levels['charm_adjusted_gamma']
        
        # Method 6: Active Hedging Detection Method
        # Identifies strikes with high gamma and high gamma-weighted volume
        self.data['active_hedging'] = self.data['gxoi_zscore'] * self.data['gxvolm_zscore']
        hedging_threshold = self.data['active_hedging'].quantile(0.9)
        hedging_levels = self.data[self.data['active_hedging'] > hedging_threshold].copy()
        hedging_levels['method'] = 'Active Hedging Detection'
        hedging_levels['score'] = hedging_levels['active_hedging']
        
        # Method 7: Value-Volume Divergence Method
        # Identifies strikes where value and volume flows diverge significantly
        self.data['value_volume_divergence'] = (
            self.data['value_bs_zscore'] - self.data['volmbs_15m_zscore']
        ).abs()
        value_threshold = self.data['value_volume_divergence'].quantile(0.9)
        value_levels = self.data[self.data['value_volume_divergence'] > value_threshold].copy()
        value_levels['method'] = 'Value-Volume Divergence'
        value_levels['score'] = value_levels['value_volume_divergence']
        
        # Combine all identified levels
        all_levels = pd.concat([
            gamma_levels, divergence_levels, flow_levels, vol_levels, 
            charm_levels, hedging_levels, value_levels
        ])
        
        # Keep only essential columns for the output
        columns_to_keep = ['date', 'strike', 'method', 'score', 'gxoi', 'dxoi', 
                          'volmbs_15m', 'charmxoi', 'vannaxoi', 'vommaxoi', 'gxvolm', 'value_bs']
        self.darkpool_levels = all_levels[columns_to_keep].sort_values('score', ascending=False)
        
        print(f"Identified {len(self.darkpool_levels)} potential darkpool levels using 7 methods")
        return self.darkpool_levels
    
    def rank_top_levels(self, n=7):
        """
        Rank and select the top n darkpool levels
        
        Parameters:
        -----------
        n : int
            Number of top levels to select (default: 7)
        """
        if len(self.darkpool_levels) == 0:
            print("No darkpool levels identified. Please run identify_darkpool_levels first.")
            return []
        
        # Group by strike and aggregate scores across methods
        strike_scores = self.darkpool_levels.groupby('strike').agg({
            'score': 'sum',
            'method': lambda x: ', '.join(set(x)),
            'gxoi': 'mean',
            'dxoi': 'mean',
            'volmbs_15m': 'mean',
            'charmxoi': 'mean',
            'vannaxoi': 'mean',
            'vommaxoi': 'mean',
            'gxvolm': 'mean',
            'value_bs': 'mean'
        }).reset_index()
        
        # Sort by aggregated score and select top n
        self.top_levels = strike_scores.sort_values('score', ascending=False).head(n)
        
        print(f"Selected top {n} darkpool levels")
        return self.top_levels
    
    def refine_to_ultra_levels(self, n=3):
        """
        Further refine to the ultra n levels with highest plausibility
        
        Parameters:
        -----------
        n : int
            Number of ultra levels to select (default: 3)
        """
        if len(self.top_levels) == 0:
            print("No top levels selected. Please run rank_top_levels first.")
            return []
        
        # Calculate a composite plausibility score based on multiple factors
        self.top_levels['plausibility'] = (
            # Factor 1: Normalized gamma concentration
            self.top_levels['gxoi'].abs() / self.top_levels['gxoi'].abs().max() +
            # Factor 2: Normalized delta-gamma alignment
            (1 - (self.top_levels['gxoi'] * self.top_levels['dxoi']).abs() / 
             ((self.top_levels['gxoi'].abs() * self.top_levels['dxoi'].abs()).replace(0, 1))) +
            # Factor 3: Normalized flow consistency
            self.top_levels['volmbs_15m'].abs() / self.top_levels['volmbs_15m'].abs().max() +
            # Factor 4: Normalized charm effect
            self.top_levels['charmxoi'].abs() / self.top_levels['charmxoi'].abs().max() +
            # Factor 5: Method diversity (more methods = higher plausibility)
            self.top_levels['method'].apply(lambda x: len(x.split(', '))) / 7
        )
        
        # Select the ultra levels with highest plausibility
        ultra_levels = self.top_levels.sort_values('plausibility', ascending=False).head(n)
        
        print(f"Refined to {n} ultra darkpool levels with highest plausibility")
        return ultra_levels
    
    def visualize_levels(self, levels=None, filename="darkpool_levels.png"):
        """
        Visualize the identified darkpool levels
        
        Parameters:
        -----------
        levels : DataFrame
            The levels to visualize (default: self.top_levels)
        filename : str
            Output filename for the visualization (default: "darkpool_levels.png")
        """
        if levels is None:
            if len(self.top_levels) > 0:
                levels = self.top_levels
            elif len(self.darkpool_levels) > 0:
                levels = self.darkpool_levels.drop_duplicates('strike').head(10)
            else:
                print("No levels to visualize")
                return
        
        plt.figure(figsize=(12, 8))
        
        # Plot strikes on x-axis and normalized metrics on y-axis
        metrics = ['gxoi', 'dxoi', 'volmbs_15m', 'charmxoi', 'vannaxoi', 'vommaxoi', 'gxvolm', 'value_bs']
        for metric in metrics:
            # Normalize the metric for visualization
            normalized = levels[metric] / levels[metric].abs().max()
            plt.plot(levels['strike'], normalized, 'o-', label=metric)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Strike Price')
        plt.ylabel('Normalized Metric Value')
        plt.title(f'Darkpool Level Analysis for {self.symbol}')
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join("output", filename))
        print(f"Visualization saved as {filename}")
        
        return os.path.join("output", filename)
    
    def generate_report(self, filename="darkpool_analysis_report.md"):
        """
        Generate a comprehensive report of the analysis
        
        Parameters:
        -----------
        filename : str
            Output filename for the report (default: "darkpool_analysis_report.md")
        """
        if len(self.darkpool_levels) == 0:
            print("No analysis results to report. Please run the analysis first.")
            return
        
        report = []
        report.append(f"# Darkpool Analysis Report for {self.symbol}")
        report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        report.append(f"\nTimeframe: Past {self.days_back} days")
        
        report.append("\n## Methodology Overview")
        report.append("\nThis analysis uses seven distinct methodologies to identify potential darkpool levels:")
        
        methodologies = {
            "High Gamma Imbalance": "Identifies strikes with unusually high gamma concentration, indicating potential dealer hedging needs that could be exploited by darkpool participants.",
            "Delta-Gamma Divergence": "Identifies strikes where delta and gamma imbalances diverge significantly, suggesting complex positioning that may involve darkpool activity.",
            "Flow Anomaly": "Identifies strikes with unusual flow patterns across timeframes, potentially indicating staggered darkpool execution.",
            "Volatility Sensitivity": "Identifies strikes with high vanna and vomma exposure, suggesting volatility-based darkpool strategies.",
            "Charm-Adjusted Gamma": "Identifies strikes with high gamma that are also sensitive to time decay, indicating potential expiration-related darkpool positioning.",
            "Active Hedging Detection": "Identifies strikes with high gamma and high gamma-weighted volume, suggesting active hedging that may be related to darkpool execution.",
            "Value-Volume Divergence": "Identifies strikes where value and volume flows diverge significantly, potentially indicating large darkpool participants entering positions."
        }
        
        for method, description in methodologies.items():
            report.append(f"\n### {method}")
            report.append(f"\n{description}")
        
        report.append("\n## Identified Darkpool Levels")
        report.append("\nThe following levels were identified as potential darkpool activity zones:")
        
        # Format the top levels as a markdown table
        if len(self.top_levels) > 0:
            report.append("\n| Strike | Methods | Gamma Concentration | Delta Exposure | Flow (15m) | Charm Effect | Vanna Effect | Vomma Effect | Active Hedging | Value Flow |")
            report.append("|--------|---------|---------------------|---------------|------------|-------------|-------------|-------------|---------------|------------|")
            
            for _, row in self.top_levels.iterrows():
                report.append(f"| {row['strike']} | {row['method']} | {row['gxoi']:.2f} | {row['dxoi']:.2f} | {row['volmbs_15m']:.2f} | {row['charmxoi']:.2f} | {row['vannaxoi']:.2f} | {row['vommaxoi']:.2f} | {row['gxvolm']:.2f} | {row['value_bs']:.2f} |")
        
        # Add ultra levels if available
        if 'plausibility' in self.top_levels.columns:
            ultra_levels = self.top_levels.sort_values('plausibility', ascending=False).head(3)
            
            report.append("\n## Ultra Darkpool Levels")
            report.append("\nThe following three levels have the highest plausibility of significant darkpool activity:")
            
            report.append("\n| Strike | Plausibility | Methods | Gamma Concentration | Delta Exposure | Flow (15m) | Charm Effect |")
            report.append("|--------|--------------|---------|---------------------|---------------|------------|-------------|")
            
            for _, row in ultra_levels.iterrows():
                report.append(f"| {row['strike']} | {row['plausibility']:.4f} | {row['method']} | {row['gxoi']:.2f} | {row['dxoi']:.2f} | {row['volmbs_15m']:.2f} | {row['charmxoi']:.2f} |")
            
            report.append("\n## Methodology Relationships")
            report.append("\nThe three ultra darkpool levels were identified through a composite analysis that considers the relationships between different methodologies:")
            
            report.append("\n1. **Gamma-Delta Relationship**: High gamma concentration coupled with significant delta exposure indicates potential dealer hedging needs that darkpool participants can exploit.")
            report.append("\n2. **Flow-Gamma Relationship**: Unusual flow patterns at strikes with high gamma concentration suggest darkpool participants may be positioning around key hedging levels.")
            report.append("\n3. **Volatility-Time Decay Relationship**: The interaction between vanna, vomma, and charm effects can reveal complex darkpool strategies that exploit volatility regime changes and time decay.")
            
            report.append("\n## Conclusion")
            report.append("\nThe identified ultra darkpool levels represent the most plausible zones of significant darkpool activity based on a comprehensive analysis of ConvexValue metrics. These levels can serve as key support/resistance zones and may be particularly important for understanding market structure and potential price action.")
        
        # Write the report to file
        with open(os.path.join("output", filename), 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report generated and saved as {filename}")
        return os.path.join("output", filename)

# Example usage
if __name__ == "__main__":
    analyzer = DarkpoolAnalyzer(symbol="SPY", days_back=5)
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.identify_darkpool_levels()
    analyzer.rank_top_levels(n=7)
    ultra_levels = analyzer.refine_to_ultra_levels(n=3)
    analyzer.visualize_levels(ultra_levels, "ultra_darkpool_levels.png")
    analyzer.generate_report()
