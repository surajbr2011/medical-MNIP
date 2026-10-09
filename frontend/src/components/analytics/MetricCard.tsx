import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  colorClass: string;
}

export default function MetricCard({ title, value, colorClass }: MetricCardProps) {
  return (
    <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl transition-all duration-500">
      <h3 className="text-sm font-medium text-gray-400 mb-1">{title}</h3>
      <div className="text-3xl font-bold text-white mb-2 transition-all duration-300 transform">
        {value}
      </div>
      <div className={`text-sm ${colorClass}`}>Real-time</div>
    </div>
  );
}
