import React from 'react';
import { TradingSidebar } from '@/components/TradingSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { DotPlotChart } from '@/components/charts/DotPlotChart';
import { HistogramChart } from '@/components/charts/HistogramChart';
import { HorizontalHeatmapGrid } from '@/components/charts/HorizontalHeatmapGrid';

const TradingPage = () => {
  return (
    <div className="grid-dashboard bg-[var(--bg-primary)] text-[var(--text-primary)]">
      <TradingSidebar />
      <div className="flex flex-col min-h-screen">
        <DashboardHeader />
        <main className="flex-1 p-6 space-y-6 animate-fade-in">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="heading-primary text-3xl mb-2">Trading Analytics</h1>
            <p className="text-[var(--text-muted)]">
              Visualize market activity and trade performance.
            </p>
          </div>

          {/* Strike-wise Horizontal Heatmap Example */}
          <section>
            <div className="panel-base p-6 h-[500px]">
              <h2 className="heading-secondary mb-4">
                Strike-wise Horizontal Heatmap Example
              </h2>
              <div className="h-[420px] w-full">
                {(() => {
                  // 14 rows, 14 columns, spectrum values
                  const xLabels = Array.from({ length: 14 }, (_, i) =>
                    String.fromCharCode(110 + i)
                  ); // 'n' to 'z'
                  const yLabels = Array.from({ length: 13 }, (_, i) =>
                    String.fromCharCode(97 + i)
                  ); // 'a' to 'm'
                  // Fill: center row is 0, interpolate up (+7) and down (-7)
                  const data = yLabels.map((_, rowIdx) =>
                    xLabels.map((_, colIdx) => 7 - rowIdx)
                  );
                  return (
                    <HorizontalHeatmapGrid
                      data={data}
                      xLabels={xLabels}
                      yLabels={yLabels}
                      min={-7}
                      max={7}
                      legendLabel="value"
                    />
                  );
                })()}
              </div>
            </div>
          </section>

          {/* Charts Section */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="panel-base p-6 h-[500px]">
              <h3 className="heading-secondary mb-6">
                Asset Volatility vs. Volume (Dot Plot)
              </h3>
              <div className="h-[420px]">
                <DotPlotChart />
              </div>
            </div>
            <div className="panel-base p-6 h-[500px]">
              <h3 className="heading-secondary mb-6">
                Trade P/L Distribution (Histogram)
              </h3>
              <div className="h-[420px]">
                <HistogramChart />
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};

export default TradingPage;