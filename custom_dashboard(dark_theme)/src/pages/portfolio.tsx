
import React from 'react';
import { TradingSidebar } from '@/components/TradingSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { PortfolioValueChart } from '@/components/charts/PortfolioValueChart';
import { AssetBreakdownTable } from '@/components/tables/AssetBreakdownTable';

const Portfolio = () => {
  return (
    <div className="grid-dashboard bg-[var(--bg-primary)] text-[var(--text-primary)]">
      {/* <DebugCSSVariables /> Removed */}
      <TradingSidebar />
      
      <div className="flex flex-col min-h-screen">
        <DashboardHeader />
        
        <main className="flex-1 p-6 space-y-6 animate-fade-in">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="heading-primary text-3xl mb-2">Portfolio Overview</h1>
            <p className="text-[var(--text-muted)]">Track your investments, performance, and asset allocation.</p>
          </div>

          {/* Charts Section (Top Third) */}
          <section className="space-y-6">
            <div className="panel-base p-6">
              <h3 className="heading-secondary mb-6">Portfolio Net Value Over Time</h3>
              <PortfolioValueChart />
            </div>
            {/* Additional charts can be added here */}
          </section>

          {/* Tables Section (Bottom Two-Thirds) */}
          <section className="space-y-6 mt-8">
            <div className="panel-base p-6">
              <h3 className="heading-secondary mb-6">Asset Breakdown</h3>
              <AssetBreakdownTable />
            </div>
            {/* Additional tables can be added here */}
          </section>
        </main>
      </div>
    </div>
  );
};

export default Portfolio;