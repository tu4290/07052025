
import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  Tooltip,
} from 'recharts';

// Example data — you can pass real data later
const data = [
  { category: 'BTC', value: 12.8 },
  { category: 'ETH', value: 0.4 },
  { category: 'SOL', value: -7.6 },
  { category: 'DOGE', value: 2.4 },
  { category: 'CAKE', value: -2.8 },
  { category: 'BNB', value: -10.0 },
  { category: 'LINK', value: 6.5 },
];

// Helper to generate gradient based on value
const getBarColor = (v: number) => {
  // - Map negatives (min -10) to #ef4444 (red), near zero to #8b5cf6 (purple), positives (max +13) to #10b981 (green)
  if (v > 0.5) return 'url(#heatGradientPositive)';
  if (v < -0.5) return 'url(#heatGradientNegative)';
  return 'url(#heatGradientNeutral)';
};

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const entry = payload[0].payload;
    return (
      <div className="bg-background border border-border rounded-lg p-2 shadow">
        <div className="font-mono font-bold mb-1">{entry.category}</div>
        <div className="text-xs text-muted-foreground">
          Value: <span className={entry.value > 0 ? 'text-[var(--positive)]' : entry.value < 0 ? 'text-[var(--negative)]' : 'text-[var(--text-primary)]'}>
            {entry.value}
          </span>
        </div>
      </div>
    );
  }
  return null;
};

export const HorizontalHeatmapChart = () => {
  // Find min/max value for domain. Adjust as needed.
  const minVal = Math.min(...data.map(d => d.value), -10);
  const maxVal = Math.max(...data.map(d => d.value), 13);

  return (
    <div className="panel-base py-7 px-6 flex flex-col items-center">
      {/* Barometer/Spectrum Key */}
      <div className="w-full flex flex-col items-center mb-6 select-none">
        <div className="flex items-center w-[90%] max-w-[650px] mx-auto">
          <svg width="100%" height="22" viewBox="0 0 300 22" className="flex-1">
            <defs>
              <linearGradient id="heat-spectrum" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="50%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
            </defs>
            <rect x="0" y="8" width="300" height="6" rx="3" fill="url(#heat-spectrum)" />
          </svg>
          <div className="flex flex-row w-full absolute top-0 left-0 justify-between mt-[26px] h-5 px-[2px] pointer-events-none text-xs font-mono text-muted-foreground" style={{width:'95%'}}>
            <span>-10</span>
            <span>0</span>
            <span>+13</span>
          </div>
        </div>
        <div className="flex w-full justify-between px-2 mt-2 text-xs text-muted-foreground">
          <span>Negative</span>
          <span>Neutral</span>
          <span>Positive</span>
        </div>
      </div>
      {/* Chart */}
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={data}
            margin={{ top: 10, right: 30, left: 40, bottom: 24 }}
            barCategoryGap={16}
          >
            {/* Gradients for bars */}
            <defs>
              <linearGradient id="heatGradientPositive" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
              <linearGradient id="heatGradientNegative" x1="1" y1="0" x2="0" y2="0">
                <stop offset="0%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
              <linearGradient id="heatGradientNeutral" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="var(--border-primary)" strokeOpacity={0.24} />
            <XAxis
              type="number"
              domain={[minVal, maxVal]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}
            />
            <YAxis
              dataKey="category"
              type="category"
              axisLine={false}
              tickLine={false}
              width={60}
              tick={{ fill: 'var(--text-primary)', fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="value"
              radius={[0, 7, 7, 0]}
            >
              {data.map((entry, idx) => (
                <Cell key={`cell-${idx}`} fill={getBarColor(entry.value)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
