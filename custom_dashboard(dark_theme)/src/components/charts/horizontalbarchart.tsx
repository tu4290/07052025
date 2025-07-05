
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell, Tooltip } from 'recharts';

const data = [
  { name: 'DeFi', value: 15.8, category: 'Decentralized Finance' },
  { name: 'Gaming', value: -8.4, category: 'Gaming & Metaverse' },
  { name: 'AI/ML', value: 22.3, category: 'Artificial Intelligence' },
  { name: 'Layer 1', value: 4.7, category: 'Layer 1 Protocols' },
  { name: 'Layer 2', value: -2.1, category: 'Layer 2 Solutions' },
  { name: 'Storage', value: 9.6, category: 'Storage & Cloud' },
  { name: 'Privacy', value: -12.7, category: 'Privacy Coins' },
  { name: 'Oracles', value: 6.8, category: 'Oracle Networks' },
];

const getBarGradient = (value: number) => {
  if (value > 0) {
    if (value > 15) return 'url(#hGradientHighPositive)';
    if (value > 8) return 'url(#hGradientMediumPositive)';
    if (value > 4) return 'url(#hGradientLowPositive)';
    return 'url(#hGradientMinimalPositive)';
  } else {
    if (value < -10) return 'url(#hGradientHighNegative)';
    if (value < -5) return 'url(#hGradientMediumNegative)';
    return 'url(#hGradientLowNegative)';
  }
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isPositive = data.value > 0;
    
    return (
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-accent)] rounded-lg p-3 shadow-xl backdrop-blur-sm">
        <div className="text-[var(--text-primary)] font-medium mb-1">{data.category}</div>
        <div className={`text-mono font-bold text-lg ${isPositive ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
          {isPositive ? '+' : ''}{data.value}%
        </div>
        <div className="text-[var(--text-muted)] text-xs mt-1">Sector Performance</div>
      </div>
    );
  }
  return null;
};

// SVG tick for axis (matches style used in other charts)
const AxisTick = (props: any) => {
  const { x, y, payload } = props;
  return (
    <text
      x={x}
      y={y}
      dy={props.dy || 0}
      dx={props.dx || 0}
      textAnchor={props.textAnchor || "middle"}
      fill="#cccccc"
      fontSize={12}
      fontFamily="JetBrains Mono, monospace"
    >
      {payload.value}
    </text>
  );
};

export const HorizontalBarChart = () => {
  return (
    <div className="h-80 relative">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical" // IMPORTANT: for horizontal bars
          data={data}
          margin={{ top: 24, right: 30, left: 60, bottom: 12 }}
        >
          <defs>
            {/* Horizontal Positive Gradients */}
            <linearGradient id="hGradientHighPositive" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--accent-tertiary)" stopOpacity={0.2} />
              <stop offset="100%" stopColor="var(--accent-tertiary)" stopOpacity={0.9} />
            </linearGradient>
            <linearGradient id="hGradientMediumPositive" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity={0.2} />
              <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity={0.8} />
            </linearGradient>
            <linearGradient id="hGradientLowPositive" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--positive)" stopOpacity={0.2} />
              <stop offset="100%" stopColor="var(--positive)" stopOpacity={0.7} />
            </linearGradient>
            <linearGradient id="hGradientMinimalPositive" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--positive-dim)" stopOpacity={0.1} />
              <stop offset="100%" stopColor="var(--positive-dim)" stopOpacity={0.6} />
            </linearGradient>
            
            {/* Horizontal Negative Gradients */}
            <linearGradient id="hGradientHighNegative" x1="1" y1="0" x2="0" y2="0">
              <stop offset="0%" stopColor="var(--negative)" stopOpacity={0.8} />
              <stop offset="100%" stopColor="var(--negative)" stopOpacity={0.2} />
            </linearGradient>
            <linearGradient id="hGradientMediumNegative" x1="1" y1="0" x2="0" y2="0">
              <stop offset="0%" stopColor="var(--negative-dim)" stopOpacity={0.7} />
              <stop offset="100%" stopColor="var(--negative-dim)" stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="hGradientLowNegative" x1="1" y1="0" x2="0" y2="0">
              <stop offset="0%" stopColor="#8B5A5A" stopOpacity={0.6} />
              <stop offset="100%" stopColor="#8B5A5A" stopOpacity={0.1} />
            </linearGradient>
          </defs>

          <CartesianGrid 
            strokeDasharray="2 4" 
            stroke="var(--border-primary)" 
            strokeOpacity={0.3}
            horizontal={false}
            vertical={true}
          />
          <XAxis
            type="number"
            axisLine={false}
            tickLine={false}
            tick={<AxisTick />}
            domain={['dataMin - 5', 'dataMax + 5']}
          />
          <YAxis
            type="category"
            dataKey="name"
            axisLine={false}
            tickLine={false}
            tick={<AxisTick />}
            width={80}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            dataKey="value"
            radius={[0, 6, 6, 0]}
            style={{
              filter: 'drop-shadow(0 2px 6px rgba(0,0,0,0.15))'
            }}
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getBarGradient(entry.value)}
                className="hover:brightness-110 transition-all duration-300 ease-out"
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
