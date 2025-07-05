
import React from 'react';
import { Search, Bell, Settings, RefreshCw } from 'lucide-react';

export const DashboardHeader = () => {
  return (
    <header className="panel-base m-6 mb-0 p-4 flex items-center justify-between animate-slide-up">
      <div className="flex items-center space-x-6">
        <div>
          <h1 className="heading-primary">Trading Dashboard</h1>
          <p className="text-[var(--text-muted)] text-sm">
            Real-time market analysis • Last updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        {/* Market Status */}
        <div className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-[var(--bg-tertiary)]">
          <div className="w-2 h-2 rounded-full bg-[var(--positive)] animate-pulse" />
          <span className="text-sm font-medium text-[var(--text-primary)]">Markets Open</span>
        </div>
        
        {/* Search */}
        <button className="interactive-base p-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent-primary)]">
          <Search size={18} />
        </button>
        
        {/* Refresh */}
        <button className="interactive-base p-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent-primary)]">
          <RefreshCw size={18} />
        </button>
        
        {/* Notifications */}
        <button className="interactive-base p-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent-primary)] relative">
          <Bell size={18} />
          <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-[var(--accent-secondary)] flex items-center justify-center">
            <span className="text-xs font-bold text-white">3</span>
          </div>
        </button>
        
        {/* Settings */}
        <button className="interactive-base p-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent-primary)]">
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
};