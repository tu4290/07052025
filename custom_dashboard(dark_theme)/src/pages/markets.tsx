
import React from 'react';
import { TradingSidebar } from '@/components/TradingSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { VerticalBarChart } from '@/components/charts/VerticalBarChart';
import { HorizontalBarChart } from '@/components/charts/HorizontalBarChart';
import { GaugeChart } from '@/components/charts/GaugeChart';
import { MarketPerformanceGrid } from '@/components/charts/MarketPerformanceGrid';

const Markets = () => {
  return (
    <div className="grid-dashboard bg-[var(--bg-primary)] text-[var(--text-primary)]">
      <TradingSidebar />
      
      <div className="flex flex-col min-h-screen">
        <DashboardHeader />
        
        <main className="flex-1 p-6 space-y-6 animate-fade-in">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="heading-primary text-3xl mb-2">Market Analysis</h1>
            <p className="text-[var(--text-muted)]">Comprehensive market performance with dynamic visualizations</p>
          </div>

          {/* Enhanced Gauge Charts Row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <GaugeChart 
              title="Market Sentiment"
              value={73}
              min={0}
              max={100}
              unit="%"
            />
            <GaugeChart 
              title="Portfolio Health"
              value={-12}
              min={-50}
              max={50}
              unit="%"
            />
            <GaugeChart 
              title="Risk Level"
              value={28}
              min={0}
              max={100}
              unit="/100"
            />
            <GaugeChart 
              title="Volatility Index"
              value={-8.5}
              min={-20}
              max={20}
              unit=""
            />
          </div>

          {/* Bar Charts Grid */}
          <div className="grid grid-cols-2 gap-8">
            {/* Vertical Bar Chart */}
            <div className="panel-base p-6">
              <h3 className="heading-secondary mb-6">Top Performers (24h)</h3>
              <VerticalBarChart />
            </div>

            {/* Horizontal Bar Chart */}
            <div className="panel-base p-6">
              <h3 className="heading-secondary mb-6">Sector Performance</h3>
              <HorizontalBarChart />
            </div>
          </div>

          {/* Market Performance Grid */}
          <div className="panel-base p-6">
            <h3 className="heading-secondary mb-6">Market Heatmap Performance</h3>
            <MarketPerformanceGrid />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Markets;