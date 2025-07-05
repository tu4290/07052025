
import React from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Area, AreaChart } from 'recharts';
import { MoreHorizontal, TrendingUp, Maximize2 } from 'lucide-react';

// Mock data for demonstration
const priceData = [
  { time: '09:00', price: 42580, volume: 1200 },
  { time: '10:00', price: 42750, volume: 1850 },
  { time: '11:00', price: 42980, volume: 2100 },
  { time: '12:00', price: 42850, volume: 1650 },
  { time: '13:00', price: 43120, volume: 2350 },
  { time: '14:00', price: 43450, volume: 1900 },
  { time: '15:00', price: 43280, volume: 1750 },
  { time: '16:00', price: 43680, volume: 2200 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-accent)] rounded-lg p-3 shadow-xl backdrop-blur-sm text-[var(--text-primary)]">
        <p className="text-[var(--text-muted)] mb-1">{label}</p>
        <p className="text-[var(--text-primary)] font-medium text-mono">
          ${payload[0].value.toLocaleString()}
        </p>
        <p className="text-[var(--accent-secondary)] text-xs">
          Vol: {payload[0].payload.volume}
        </p>
      </div>
    );
  }
  return null;
};

// Explicit SVG tick for axis using CSS variables
const AxisTick = (props: any) => {
  const { x, y, payload } = props;
  return (
    <text
      x={x}
      y={y}
      dy={props.dy || 0}
      dx={props.dx || 0}
      textAnchor={props.textAnchor || "middle"}
      fill="var(--text-muted)"
      fontSize={12}
      fontFamily="JetBrains Mono, monospace"
    >
      {payload.value}
    </text>
  );
};

export const PriceChart = () => {
  const currentPrice = priceData[priceData.length - 1].price;
  const priceChange = currentPrice - priceData[0].price;
  const priceChangePercent = ((priceChange / priceData[0].price) * 100).toFixed(2);

  return (
    <div className="chart-container animate-slide-up">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <div>
            <h2 className="heading-secondary mb-1">BTC/USDT</h2>
            <div className="flex items-center space-x-3">
              <span className="metric-medium text-mono">
                ${currentPrice.toLocaleString()}
              </span>
              <div className={`flex items-center space-x-1 ${
                priceChange >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
              }`}>
                <TrendingUp size={16} />
                <span className="font-medium">
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(0)} ({priceChangePercent}%)
                </span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button className="interactive-base p-2 rounded-lg text-[var(--text-muted)] hover:text-[var(--accent-primary)]">
            <Maximize2 size={16} />
          </button>
          <button className="interactive-base p-2 rounded-lg text-[var(--text-muted)] hover:text-[var(--accent-primary)]">
            <MoreHorizontal size={16} />
          </button>
        </div>
      </div>
      
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={priceData}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis 
              dataKey="time" 
              axisLine={false}
              tickLine={false}
              tick={<AxisTick />}
              dy={10}
            />
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={<AxisTick />}
              dx={-10}
              domain={['dataMin - 100', 'dataMax + 100']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="price"
              stroke="var(--accent-primary)"
              strokeWidth={2}
              fill="url(#priceGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      {/* Time Period Selector */}
      <div className="flex items-center justify-center space-x-1 mt-4">
        {['1H', '4H', '1D', '1W', '1M'].map((period, index) => (
          <button
            key={period}
            className={`px-3 py-1 rounded text-sm font-medium transition-all duration-150 ${
              index === 2 
                ? 'bg-[var(--accent-primary)] bg-opacity-10 text-[var(--accent-primary)]' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
            }`}
          >
            {period}
          </button>
        ))}
      </div>
    </div>
  );
};