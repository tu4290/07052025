
import React from 'react';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table'; // Using shadcn table components

interface Asset {
  symbol: string;
  name: string;
  quantity: number;
  costBasis: number;
  currentPrice: number;
  marketValue: number;
  pl: number;
  plPercent: number;
  allocation: number;
  volatility: string; // e.g., 'Low', 'Medium', 'High'
}

const mockAssets: Asset[] = [
  { symbol: 'BTC', name: 'Bitcoin', quantity: 0.5, costBasis: 30000, currentPrice: 60000, marketValue: 30000, pl: 15000, plPercent: 100, allocation: 30, volatility: 'High' },
  { symbol: 'ETH', name: 'Ethereum', quantity: 10, costBasis: 2000, currentPrice: 3500, marketValue: 35000, pl: 15000, plPercent: 75, allocation: 35, volatility: 'High' },
  { symbol: 'AAPL', name: 'Apple Inc.', quantity: 50, costBasis: 150, currentPrice: 170, marketValue: 8500, pl: 1000, plPercent: 13.33, allocation: 8.5, volatility: 'Medium' },
  { symbol: 'MSFT', name: 'Microsoft Corp.', quantity: 30, costBasis: 250, currentPrice: 420, marketValue: 12600, pl: 5100, plPercent: 68, allocation: 12.6, volatility: 'Medium' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', quantity: 5, costBasis: 2200, currentPrice: 2700, marketValue: 13500, pl: 2500, plPercent: 22.73, allocation: 13.5, volatility: 'Medium' },
  { symbol: 'USDT', name: 'Tether', quantity: 400, costBasis: 1, currentPrice: 1, marketValue: 400, pl: 0, plPercent: 0, allocation: 0.4, volatility: 'Low' },
];

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
};

const formatPercent = (value: number) => {
  return `${value.toFixed(2)}%`;
};

export const AssetBreakdownTable = () => {
  return (
    <div className="w-full overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[var(--border-secondary)]">
            <TableHead className="text-[var(--text-secondary)]">Symbol</TableHead>
            <TableHead className="text-[var(--text-secondary)]">Name</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">Quantity</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">Cost Basis</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">Current Price</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">Market Value</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">P/L</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">P/L %</TableHead>
            <TableHead className="text-[var(--text-secondary)] text-right">Allocation</TableHead>
            <TableHead className="text-[var(--text-secondary)]">Volatility</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {mockAssets.map((asset) => (
            <TableRow key={asset.symbol} className="border-[var(--border-primary)] hover:bg-[var(--bg-hover)] transition-colors duration-150">
              <TableCell className="font-medium text-[var(--text-primary)]">{asset.symbol}</TableCell>
              <TableCell className="text-[var(--text-muted)]">{asset.name}</TableCell>
              <TableCell className="text-right text-mono text-[var(--text-primary)]">{asset.quantity.toLocaleString()}</TableCell>
              <TableCell className="text-right text-mono text-[var(--text-primary)]">{formatCurrency(asset.costBasis)}</TableCell>
              <TableCell className="text-right text-mono text-[var(--text-primary)]">{formatCurrency(asset.currentPrice)}</TableCell>
              <TableCell className="text-right text-mono text-[var(--text-primary)] font-semibold">{formatCurrency(asset.marketValue)}</TableCell>
              <TableCell className={`text-right text-mono font-medium ${asset.pl >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
                {asset.pl >= 0 ? '+' : ''}{formatCurrency(asset.pl)}
              </TableCell>
              <TableCell className={`text-right text-mono ${asset.plPercent >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
                {formatPercent(asset.plPercent)}
              </TableCell>
              <TableCell className="text-right text-mono text-[var(--text-primary)]">{formatPercent(asset.allocation)}</TableCell>
              <TableCell>
                <span className={`px-2 py-1 text-xs rounded-full font-medium
                  ${asset.volatility === 'High' ? 'bg-[var(--negative-dim)] text-[var(--negative)]' : 
                    asset.volatility === 'Medium' ? 'bg-[var(--accent-secondary-dim)] text-[var(--accent-secondary)]' : 
                    'bg-[var(--positive-dim)] text-[var(--positive)]'}`}>
                  {asset.volatility}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};
