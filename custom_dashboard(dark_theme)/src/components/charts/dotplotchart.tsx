
import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ChartContainer, ChartTooltipContent, ChartLegend, ChartLegendContent, type ChartConfig } from '@/components/ui/chart';

const dotPlotData = [
  { asset: 'BTC', volatility: 2.5, volume: 1500000, marketCap: 1.2e12 },
  { asset: 'ETH', volatility: 3.1, volume: 1200000, marketCap: 4.5e11 },
  { asset: 'SOL', volatility: 4.5, volume: 900000, marketCap: 8.0e10 },
  { asset: 'ADA', volatility: 2.8, volume: 750000, marketCap: 2.0e10 },
  { asset: 'DOGE', volatility: 5.2, volume: 1800000, marketCap: 2.5e10 },
  { asset: 'LINK', volatility: 3.5, volume: 600000, marketCap: 1.0e10 },
];

const chartConfig: ChartConfig = {
  volume: {
    label: "Volume (USD)",
    color: "var(--accent-primary)",
  },
  volatility: {
    label: "Volatility (%)",
  },
  marketCap: {
    label: "Market Cap",
  },
  btc: { label: "BTC", color: "var(--accent-primary)" },
  eth: { label: "ETH", color: "var(--accent-secondary)" },
  sol: { label: "SOL", color: "var(--positive)" },
  ada: { label: "ADA", color: "var(--accent-tertiary)" },
  doge: { label: "DOGE", color: "var(--negative)" },
  link: { label: "LINK", color: "var(--neutral)" },
};

// Define a specific type for asset keys that are guaranteed to have a color in chartConfig
type AssetKeyWithColor = 'btc' | 'eth' | 'sol' | 'ada' | 'doge' | 'link';

const CustomTooltipDotPlot = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded-lg p-3 shadow-xl backdrop-blur-sm text-[var(--text-primary)]">
        <p className="font-bold text-lg">{data.asset}</p>
        <p>Volatility: <span className="font-medium">{data.volatility}%</span></p>
        <p>Volume: <span className="font-medium">${data.volume.toLocaleString()}</span></p>
        <p>Market Cap: <span className="font-medium">${(data.marketCap / 1e9).toFixed(1)}B</span></p>
      </div>
    );
  }
  return null;
};

export const DotPlotChart = () => {
  return (
    <div className="h-96 w-full">
      <ChartContainer config={chartConfig} className="h-full w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{
              top: 20,
              right: 20,
              bottom: 20,
              left: 20,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" vertical={false}/>
            <XAxis 
              type="number" 
              dataKey="volatility" 
              name="Volatility" 
              unit="%" 
              tickLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              axisLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
              tickMargin={8}
            />
            <YAxis 
              type="number" 
              dataKey="volume" 
              name="Volume" 
              unit="$" 
              tickFormatter={(value) => `$${value / 1000000}M`}
              tickLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              axisLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
              tickMargin={8}
            />
            <ZAxis type="number" dataKey="marketCap" range={[50, 500]} name="Market Cap" unit="USD" />
            <Tooltip 
              cursor={{ strokeDasharray: '3 3', stroke: 'var(--accent-primary)' }} 
              content={<CustomTooltipDotPlot />}
              wrapperStyle={{ outline: 'none' }}
            />
            <Legend content={<ChartLegendContent />} />
            {dotPlotData.map((entry) => {
              const assetKey = entry.asset.toLowerCase() as AssetKeyWithColor;
              const configEntry = chartConfig[assetKey]; 
              
              return (
                <Scatter 
                  key={entry.asset}
                  name={entry.asset} 
                  data={[entry]} 
                  fill={configEntry.color || "var(--neutral)"}
                  shape="circle" 
                />
              );
            })}
          </ScatterChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
};
