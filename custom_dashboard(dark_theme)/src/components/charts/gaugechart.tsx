
import React from 'react';

interface GaugeChartProps {
  title: string;
  value: number;
  min: number;
  max: number;
  unit: string;
}

export const GaugeChart: React.FC<GaugeChartProps> = ({ title, value, min, max, unit }) => {
  // Calculate percentage and angle
  const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const angle = (percentage / 100) * 180 - 90; // -90 to 90 degrees
  
  // Determine colors and glow based on value
  const getGaugeStyles = () => {
    const normalizedValue = (value - min) / (max - min);
    
    if (normalizedValue > 0.7) {
      return {
        arcColor: '#10b981',
        glowColor: 'rgba(16, 185, 129, 0.6)',
        needleColor: '#059669',
        textColor: '#10b981',
        shadowColor: 'rgba(16, 185, 129, 0.3)',
        bgArcColor: 'rgba(16, 185, 129, 0.1)'
      };
    } else if (normalizedValue < 0.3) {
      return {
        arcColor: '#ef4444',
        glowColor: 'rgba(239, 68, 68, 0.6)',
        needleColor: '#dc2626',
        textColor: '#ef4444',
        shadowColor: 'rgba(239, 68, 68, 0.3)',
        bgArcColor: 'rgba(239, 68, 68, 0.1)'
      };
    } else {
      return {
        arcColor: '#8b5cf6',
        glowColor: 'rgba(139, 92, 246, 0.6)',
        needleColor: '#7c3aed',
        textColor: '#8b5cf6',
        shadowColor: 'rgba(139, 92, 246, 0.3)',
        bgArcColor: 'rgba(139, 92, 246, 0.1)'
      };
    }
  };

  const styles = getGaugeStyles();
  const isPositive = value >= 0;

  // For the barometer/spectrum
  const spectrumId = `spectrum-gradient-${title.replace(/\s/g, "")}`;

  return (
    <div className="panel-base interactive-base py-7 px-6 text-center group relative overflow-hidden flex flex-col items-center h-full">
      {/* Background gradient effect */}
      <div 
        className="absolute inset-0 opacity-5 bg-gradient-to-br from-transparent via-current to-transparent pointer-events-none"
        style={{ color: styles.arcColor }}
      />

      {/* Top Title */}
      <h4 className="text-[var(--text-secondary)] text-sm font-semibold mb-4 group-hover:text-[var(--text-primary)] transition-colors duration-300 tracking-wide uppercase mt-2">
        {title}
      </h4>
      
      {/* Overarching Barometer Spectrum */}
      <div className="flex justify-center w-full mb-6 mt-2 select-none pointer-events-none">
        <svg width="160" height="18" viewBox="0 0 160 18" className="overflow-visible">
          <defs>
            <linearGradient id={spectrumId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="30%" stopColor="#8b5cf6" />
              <stop offset="70%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
          </defs>
          {/* The colored spectrum bar */}
          <rect x="0" y="7" width="160" height="6" rx="3" fill={`url(#${spectrumId})`} />
          {/* Arrow pointer for current value */}
          <polygon
            points="0,0 6,7 12,0"
            fill={styles.arcColor}
            style={{
              opacity: 0.86,
              filter: `drop-shadow(0 1px 4px ${styles.glowColor})`
            }}
            transform={`translate(${(percentage / 100) * 160 - 6}, 0)`}
          />
        </svg>
      </div>

      {/* Gauge SVG (with better spacing below) */}
      <div className="relative flex justify-center mb-8 mt-1">
        <svg width="180" height="110" viewBox="0 0 180 110" className="overflow-visible">
          <defs>
            {/* Main gradient for the progress arc */}
            <linearGradient id={`gradient-${title}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={styles.arcColor} stopOpacity={0.4} />
              <stop offset="50%" stopColor={styles.arcColor} stopOpacity={1} />
              <stop offset="100%" stopColor={styles.arcColor} stopOpacity={0.4} />
            </linearGradient>
            
            {/* Background arc gradient */}
            <linearGradient id={`bg-gradient-${title}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#374151" stopOpacity={0.3} />
              <stop offset="50%" stopColor="#4b5563" stopOpacity={0.5} />
              <stop offset="100%" stopColor="#374151" stopOpacity={0.3} />
            </linearGradient>
            
            {/* Glow filter */}
            <filter id={`glow-${title}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            
            {/* Shadow filter */}
            <filter id={`shadow-${title}`} x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="3" stdDeviation="6" floodColor={styles.shadowColor} />
            </filter>
            
            {/* Needle glow filter */}
            <filter id={`needle-glow-${title}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          {/* Outer ring decoration */}
          <circle
            cx="90"
            cy="85"
            r="68"
            fill="none"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth="1"
          />
          
          {/* Background arc with gradient */}
          <path
            d="M 30 85 A 60 60 0 0 1 150 85"
            fill="none"
            stroke={`url(#bg-gradient-${title})`}
            strokeWidth="12"
            strokeLinecap="round"
          />
          
          {/* Progress arc with enhanced glow */}
          <path
            d="M 30 85 A 60 60 0 0 1 150 85"
            fill="none"
            stroke={`url(#gradient-${title})`}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${(percentage / 100) * 188.5} 188.5`}
            filter={`url(#glow-${title})`}
            className="transition-all duration-1000 ease-out"
          />
          
          {/* Tick marks */}
          {[0, 25, 50, 75, 100].map((tick, index) => {
            const tickAngle = (tick / 100) * 180 - 90;
            const tickX1 = 90 + 55 * Math.cos((tickAngle * Math.PI) / 180);
            const tickY1 = 85 + 55 * Math.sin((tickAngle * Math.PI) / 180);
            const tickX2 = 90 + 50 * Math.cos((tickAngle * Math.PI) / 180);
            const tickY2 = 85 + 50 * Math.sin((tickAngle * Math.PI) / 180);
            
            return (
              <line
                key={index}
                x1={tickX1}
                y1={tickY1}
                x2={tickX2}
                y2={tickY2}
                stroke="rgba(255, 255, 255, 0.3)"
                strokeWidth="2"
                strokeLinecap="round"
              />
            );
          })}
          
          {/* Center hub with enhanced styling */}
          <circle
            cx="90"
            cy="85"
            r="8"
            fill={styles.needleColor}
            filter={`url(#shadow-${title})`}
          />
          
          <circle
            cx="90"
            cy="85"
            r="6"
            fill="rgba(255, 255, 255, 0.2)"
          />
          
          <circle
            cx="90"
            cy="85"
            r="3"
            fill={styles.needleColor}
          />
          
          {/* Enhanced needle */}
          <path
            d="M 90 85 L 88 35 L 90 30 L 92 35 Z"
            fill={styles.needleColor}
            transform={`rotate(${angle} 90 85)`}
            className="transition-transform duration-1000 ease-out"
            filter={`url(#needle-glow-${title})`}
          />
          
          {/* Needle highlight */}
          <path
            d="M 90 85 L 89 35 L 90 30 L 91 35 Z"
            fill="rgba(255, 255, 255, 0.4)"
            transform={`rotate(${angle} 90 85)`}
            className="transition-transform duration-1000 ease-out"
          />
          
          {/* Value indicator dot */}
          <circle
            cx="90"
            cy="30"
            r="3"
            fill={styles.arcColor}
            transform={`rotate(${angle} 90 85)`}
            className="transition-transform duration-1000 ease-out"
            style={{ filter: `drop-shadow(0 0 6px ${styles.glowColor})` }}
          />
        </svg>
      </div>
      
      {/* Value and range */}
      <div className="space-y-3 relative z-10 mt-2">
        <div 
          className="text-3xl font-bold text-mono tracking-tight mb-1"
          style={{ color: styles.textColor }}
        >
          {isPositive ? '+' : ''}{value}{unit}
        </div>
        <div className="text-[var(--text-muted)] text-xs font-medium">
          Range: {min} to {max}{unit}
        </div>
        
        {/* Progress indicator */}
        <div className="flex items-center justify-center space-x-2 mt-4">
          <div className="w-16 h-1 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full rounded-full transition-all duration-1000 ease-out"
              style={{ 
                width: `${percentage}%`,
                backgroundColor: styles.arcColor,
                boxShadow: `0 0 8px ${styles.glowColor}`
              }}
            />
          </div>
          <span className="text-xs text-[var(--text-muted)] font-medium">
            {percentage.toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
};
