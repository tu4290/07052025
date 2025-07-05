
import React from 'react';
import { BarChart3, TrendingUp, PieChart, Settings, Bell, User, Wallet, Activity } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';

const navigationItems = [
  { icon: BarChart3, label: 'Dashboard', path: '/' },
  { icon: TrendingUp, label: 'Markets', path: '/markets' },
  { icon: PieChart, label: 'Portfolio', path: '/portfolio' },
  { icon: Activity, label: 'Trading', path: '/trading' },
  { icon: Wallet, label: 'Wallet', path: '/wallet' },
];

const settingsItems = [
  { icon: Bell, label: 'Alerts', path: '/alerts' },
  { icon: Settings, label: 'Settings', path: '/settings' },
  { icon: User, label: 'Profile', path: '/profile' },
];

export const TradingSidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  return (
    <div className="panel-base h-screen flex flex-col p-6">
      {/* Logo/Brand */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] bg-clip-text text-transparent">
          TradePro
        </h1>
        <p className="text-[var(--text-muted)] text-sm mt-1">Professional Trading</p>
      </div>
      
      {/* Main Navigation */}
      <nav className="flex-1 space-y-2">
        <div className="mb-6">
          <h3 className="text-[var(--text-muted)] text-xs uppercase tracking-wider font-medium mb-3">
            Trading
          </h3>
          {navigationItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <button
                key={item.label}
                onClick={() => handleNavigation(item.path)}
                className={`
                  w-full flex items-center px-4 py-3 rounded-lg text-left
                  transition-all duration-150 ease-out group
                  ${isActive 
                    ? 'bg-[var(--accent-primary)] bg-opacity-10 text-[var(--accent-primary)] border border-[var(--accent-primary)] border-opacity-20' 
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
                  }
                `}
              >
                <item.icon 
                  size={18} 
                  className={`mr-3 transition-colors ${
                    isActive ? 'text-[var(--accent-primary)]' : 'group-hover:text-[var(--accent-primary)]'
                  }`} 
                />
                <span className="font-medium">{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-2 h-2 rounded-full bg-[var(--accent-primary)]" />
                )}
              </button>
            );
          })}
        </div>
        
        {/* Settings Section */}
        <div className="pt-4 border-t border-[var(--border-primary)]">
          <h3 className="text-[var(--text-muted)] text-xs uppercase tracking-wider font-medium mb-3">
            Account
          </h3>
          {settingsItems.map((item) => (
            <button
              key={item.label}
              onClick={() => handleNavigation(item.path)}
              className="w-full flex items-center px-4 py-3 rounded-lg text-left text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all duration-150 ease-out group"
            >
              <item.icon size={18} className="mr-3 group-hover:text-[var(--accent-primary)]" />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </div>
      </nav>
      
      {/* User Profile */}
      <div className="mt-6 p-4 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border-secondary)]">
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-secondary)] flex items-center justify-center">
            <User size={16} className="text-white" />
          </div>
          <div className="ml-3">
            <p className="text-[var(--text-primary)] font-medium text-sm">Alex Trader</p>
            <p className="text-[var(--text-muted)] text-xs">Premium Account</p>
          </div>
        </div>
      </div>
    </div>
  );
};
