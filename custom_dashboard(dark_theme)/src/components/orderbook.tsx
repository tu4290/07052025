
import React from 'react';
import { MoreVertical } from 'lucide-react';

// Mock order book data
const asks = [
  { price: 43720, size: 0.5420, total: 2.8950 },
  { price: 43715, size: 1.2340, total: 2.3530 },
  { price: 43710, size: 0.8760, total: 1.1190 },
  { price: 43705, size: 0.2430, total: 0.2430 },
];

const bids = [
  { price: 43695, size: 0.3210, total: 0.3210 },
  { price: 43690, size: 0.9870, total: 1.3080 },
  { price: 43685, size: 1.5430, total: 2.8510 },
  { price: 43680, size: 0.7650, total: 3.6160 },
];

const OrderRow = ({ order, type }: { order: any, type: 'ask' | 'bid' }) => {
  const bgIntensity = (order.total / 4) * 100; // Normalize to percentage
  
  return (
    <div className="relative flex items-center justify-between py-1 px-2 text-sm hover:bg-[var(--bg-hover)] transition-colors">
      {/* Background depth indicator */}
      <div 
        className={`absolute left-0 top-0 h-full ${
          type === 'ask' ? 'bg-[var(--negative)]' : 'bg-[var(--positive)]'
        } opacity-10`}
        style={{ width: `${bgIntensity}%` }}
      />
      
      <span className={`text-mono font-medium relative z-10 ${
        type === 'ask' ? 'text-[var(--negative)]' : 'text-[var(--positive)]'
      }`}>
        ${order.price.toLocaleString()}
      </span>
      <span className="text-[var(--text-secondary)] text-mono relative z-10">
        {order.size.toFixed(4)}
      </span>
      <span className="text-[var(--text-muted)] text-mono text-xs relative z-10">
        {order.total.toFixed(4)}
      </span>
    </div>
  );
};

export const OrderBook = () => {
  const spread = asks[asks.length - 1].price - bids[0].price;
  const spreadPercent = ((spread / bids[0].price) * 100).toFixed(3);

  return (
    <div className="panel-base p-4 animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <h2 className="heading-secondary">Order Book</h2>
        <button className="interactive-base p-1 rounded text-[var(--text-muted)] hover:text-[var(--accent-primary)]">
          <MoreVertical size={16} />
        </button>
      </div>
      
      {/* Header */}
      <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-2 px-2">
        <span>Price (USDT)</span>
        <span>Size (BTC)</span>
        <span>Total</span>
      </div>
      
      {/* Asks (Sell Orders) */}
      <div className="space-y-0.5 mb-3">
        {asks.reverse().map((ask, index) => (
          <OrderRow key={index} order={ask} type="ask" />
        ))}
      </div>
      
      {/* Spread */}
      <div className="flex items-center justify-center py-2 mb-3 bg-[var(--bg-tertiary)] rounded-lg">
        <div className="text-center">
          <div className="text-[var(--accent-primary)] font-medium text-mono">
            ${spread.toFixed(0)}
          </div>
          <div className="text-[var(--text-muted)] text-xs">
            Spread ({spreadPercent}%)
          </div>
        </div>
      </div>
      
      {/* Bids (Buy Orders) */}
      <div className="space-y-0.5">
        {bids.map((bid, index) => (
          <OrderRow key={index} order={bid} type="bid" />
        ))}
      </div>
      
      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-[var(--border-primary)] grid grid-cols-2 gap-4 text-center">
        <div>
          <div className="text-[var(--positive)] font-medium text-mono">
            {bids.reduce((sum, bid) => sum + bid.size, 0).toFixed(4)}
          </div>
          <div className="text-[var(--text-muted)] text-xs">Total Bids</div>
        </div>
        <div>
          <div className="text-[var(--negative)] font-medium text-mono">
            {asks.reduce((sum, ask) => sum + ask.size, 0).toFixed(4)}
          </div>
          <div className="text-[var(--text-muted)] text-xs">Total Asks</div>
        </div>
      </div>
    </div>
  );
};
