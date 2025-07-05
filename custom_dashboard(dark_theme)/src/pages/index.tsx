
import React from 'react';
import { TradingSidebar } from '@/components/TradingSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { MetricsOverview } from '@/components/MetricsOverview';
import { PriceChart } from '@/components/PriceChart';
import { MarketHeatmap } from '@/components/MarketHeatmap';
import { OrderBook } from '@/components/OrderBook';
import { RecentTrades } from '@/components/RecentTrades';

const Index = () => {
  return (
    <div className="grid-dashboard bg-[var(--bg-primary)] text-[var(--text-primary)]">
      {/* Sidebar Navigation */}
      <TradingSidebar />
      
      {/* Main Content Area */}
      <div className="flex flex-col min-h-screen">
        <DashboardHeader />
        
        <main className="flex-1 p-6 space-y-6 animate-fade-in">
          {/* Key Metrics Overview */}
          <MetricsOverview />
          
          {/* Primary Trading Interface */}
          <div className="grid grid-cols-12 gap-6">
            {/* Main Price Chart */}
            <div className="col-span-8">
              <PriceChart />
            </div>
            
            {/* Order Book */}
            <div className="col-span-4">
              <OrderBook />
            </div>
          </div>
          
          {/* Secondary Analysis Tools */}
          <div className="grid grid-cols-12 gap-6">
            {/* Market Heatmap */}
            <div className="col-span-7">
              <MarketHeatmap />
            </div>
            
            {/* Recent Trades */}
            <div className="col-span-5">
              <RecentTrades />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Index;
