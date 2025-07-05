
import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Activity, BarChart3, PieChart } from 'lucide-react';

const metrics = [
  {
    label: 'Portfolio Value',
    value: '$124,573.92',
    change: '+2.34%',
    changeValue: '+$2,847.23',
    positive: true,
    icon: DollarSign,
  },
  {
    label: 'Day P&L',
    value: '$1,247.83',
    change: '+1.67%',
    changeValue: '+$20.45',
    positive: true,
    icon: TrendingUp,
  },
  {
    label: 'Active Positions',
    value: '12',
    change: '+2',
    changeValue: 'vs yesterday',
    positive: true,
    icon: Activity,
  },
  {
    label: 'Win Rate',
    value: '73.2%',
    change: '+4.1%',
    changeValue: 'this week',
    positive: true,
    icon: BarChart3,
  },
];

export const MetricsOverview = () => {
  return (
    <div className="grid-cards animate-fade-in">
      {metrics.map((metric, index) => (
        <div 
          key={metric.label}
          className="panel-base interactive-base p-6"
          style={{
            animationDelay: `${index * 100}ms`
          }}
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-[var(--accent-primary)] bg-opacity-10">
                <metric.icon size={20} className="text-[var(--accent-primary)]" />
              </div>
              <div>
                <h3 className="text-[var(--text-secondary)] text-sm font-medium">
                  {metric.label}
                </h3>
              </div>
            </div>
            
            <div className={`flex items-center space-x-1 text-sm ${
              metric.positive ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
            }`}>
              {metric.positive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
              <span className="font-medium">{metric.change}</span>
            </div>
          </div>
          
          <div className="space-y-1">
            <div className="metric-medium text-mono">
              {metric.value}
            </div>
            <div className="text-[var(--text-muted)] text-sm">
              {metric.changeValue}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
