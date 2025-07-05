
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { ChartContainer, ChartTooltipContent, ChartLegend, ChartLegendContent, type ChartConfig } from '@/components/ui/chart';

// Mock P/L data for trades
const pnlDataRaw = [
  -150, 200, -50, 300, 50, -200, 450, -100, 100, 250,
  -80, 120, -30, 180, -220, 320, -120, 70, -170, 280,
];

// Simple binning function
const createHistogramData = (data: number[], binSize: number) => {
  const bins: { [key: string]: number } = {};
  const minVal = Math.min(...data);
  const maxVal = Math.max(...data);

  for (const value of data) {
    const binStart = Math.floor(value / binSize) * binSize;
    const binEnd = binStart + binSize;
    const binName = `${binStart} to ${binEnd}`;
    bins[binName] = (bins[binName] || 0) + 1;
  }

  // Ensure all bins between min and max are present, even if count is 0
  const sortedBinNames = [];
  for (let i = Math.floor(minVal / binSize) * binSize; i < maxVal; i += binSize) {
      const binName = `${i} to ${i + binSize}`;
      sortedBinNames.push(binName);
      if (!bins[binName]) {
          bins[binName] = 0;
      }
  }
  
  return Object.keys(bins)
    .map(name => ({ name, count: bins[name] }))
    .sort((a, b) => parseFloat(a.name.split(' ')[0]) - parseFloat(b.name.split(' ')[0]));
};

const histogramData = createHistogramData(pnlDataRaw, 100); // Bin size of 100 USD

const chartConfig: ChartConfig = {
  count: {
    label: "Number of Trades",
    color: "var(--accent-secondary)",
  },
};

const CustomTooltipHistogram = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded-lg p-3 shadow-xl backdrop-blur-sm text-[var(--text-primary)]">
        <p className="font-medium">P/L Range: <span className="font-bold">{label}</span></p>
        <p>Number of Trades: <span className="font-bold" style={{color: "var(--accent-secondary)"}}>{payload[0].value}</span></p>
      </div>
    );
  }
  return null;
};

export const HistogramChart = () => {
  return (
    <div className="h-96 w-full">
      <ChartContainer config={chartConfig} className="h-full w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={histogramData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" vertical={false} />
            <XAxis 
              dataKey="name" 
              tickLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              axisLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              tickMargin={8}
              interval={0}
              angle={-30}
              textAnchor="end"
              tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
            />
            <YAxis 
              tickLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }} 
              axisLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              tickMargin={8}
              allowDecimals={false}
              tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
            />
            <Tooltip 
              cursor={{ fill: 'rgba(255, 184, 74, 0.2)' }}
              content={<CustomTooltipHistogram />}
              wrapperStyle={{ outline: 'none' }}
            />
            <Legend content={<ChartLegendContent verticalAlign="top" />} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} barSize={30}>
              {histogramData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill="var(--accent-secondary)" />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
};