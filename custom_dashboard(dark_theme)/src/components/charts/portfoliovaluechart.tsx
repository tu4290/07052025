import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ChartContainer, ChartTooltipContent, ChartLegendContent } from '@/components/ui/chart';

const chartData = [
  { date: '2025-01-01', portfolioValue: 10000, benchmarkValue: 10000 },
  { date: '2025-01-08', portfolioValue: 10200, benchmarkValue: 10100 },
  { date: '2025-01-15', portfolioValue: 10150, benchmarkValue: 10150 },
  { date: '2025-01-22', portfolioValue: 10500, benchmarkValue: 10300 },
  { date: '2025-01-29', portfolioValue: 10800, benchmarkValue: 10450 },
  { date: '2025-02-05', portfolioValue: 11000, benchmarkValue: 10600 },
  { date: '2025-02-12', portfolioValue: 10900, benchmarkValue: 10700 },
  { date: '2025-02-19', portfolioValue: 11250, benchmarkValue: 10850 },
  { date: '2025-02-26', portfolioValue: 11500, benchmarkValue: 11000 },
];

const chartConfig = {
  portfolioValue: {
    label: "Portfolio Value",
    color: "var(--accent-primary)",
  },
  benchmarkValue: {
    label: "Benchmark",
    color: "var(--accent-secondary)",
  },
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded-lg p-3 shadow-xl backdrop-blur-sm text-[var(--text-primary)]">
        <p className="text-[var(--text-muted)] mb-1">{new Date(label).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</p>
        {payload.map((item: any, index: number) => (
          <p key={index} className="font-medium text-mono" style={{ color: item.color }}>
            {item.name}: ${item.value.toLocaleString()}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export const PortfolioValueChart = () => {
  return (
    <div className="h-80 w-full">
      <ChartContainer config={chartConfig} className="h-full w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{
              top: 5,
              right: 20,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              axisLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              tickMargin={8}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
            />
            <YAxis
              tickLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              axisLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
              tickMargin={8}
              tickFormatter={(value) => `$${value / 1000}k`}
              tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
            />
            <Tooltip
              cursor={{ stroke: 'var(--accent-primary)', strokeWidth: 1, strokeDasharray: '3 3' }}
              content={<CustomTooltip />}
              wrapperStyle={{ outline: 'none' }}
            />
            <Legend content={<ChartLegendContent />} />
            <Line
              type="monotone"
              dataKey="portfolioValue"
              stroke="var(--accent-primary)"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, style: { fill: 'var(--accent-primary)', stroke: 'var(--bg-secondary)' } }}
              name="Portfolio Value"
              animationDuration={700}
            />
            <Line
              type="monotone"
              dataKey="benchmarkValue"
              stroke="var(--accent-secondary)"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              activeDot={{ r: 6, style: { fill: 'var(--accent-secondary)', stroke: 'var(--bg-secondary)' } }}
              name="Benchmark"
              animationDuration={700}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
};
