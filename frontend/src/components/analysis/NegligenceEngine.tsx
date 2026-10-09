import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { NegligenceResult, TokenAttribution } from '../../types';

interface NegligenceEngineProps {
  detection: NegligenceResult;
}

export default function NegligenceEngine({ detection }: NegligenceEngineProps) {
  const chartData = Object.entries(detection.categories || {})
    .map(([name, val]) => ({ name, value: val }))
    .sort((a, b) => a.value - b.value);

  const renderTokenHeatmap = (tokens: TokenAttribution[]) => {
    return tokens.map((ta, idx) => {
      let token = ta.token;
      const score = ta.attribution;
      const hue = 240 - Math.floor(score * 240);
      const color = `hsl(${hue}, 85%, 75%)`;
      
      if (token.startsWith("##")) {
        token = token.substring(2);
      } else {
        token = " " + token;
      }
      
      return (
        <span 
          key={idx} 
          style={{ backgroundColor: color }} 
          className="inline-block px-1 py-0.5 m-0.5 rounded text-black font-mono text-sm"
        >
          {token}
        </span>
      );
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
        <h2 className="text-xl font-semibold mb-4">Negligence Detection Engine</h2>
        
        <div className={`p-4 rounded-lg border mb-6 ${
          detection.negligent 
            ? 'bg-red-900/30 border-red-500/50 text-red-400' 
            : 'bg-green-900/30 border-green-500/50 text-green-400'
        }`}>
          <span className="font-bold">
            Negligence Flagged: {detection.negligent ? 'TRUE' : 'FALSE'}
          </span>
          <span className="ml-2">
            (Confidence: {(detection.confidence * 100).toFixed(1)}%)
          </span>
        </div>
        
        <h3 className="font-semibold mb-2">WHO ICPS Category Breakdown</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={chartData}>
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={100} tick={{ fill: '#8b949e', fontSize: 12 }} />
              <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} />
              <Bar dataKey="value" fill="#8957e5" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
        <h2 className="text-xl font-semibold mb-4">Explainability: Token Attribution Heatmap</h2>
        <div className="bg-[#1e1e1e] p-4 rounded-lg border border-[#333] leading-loose">
          {detection.token_attributions && detection.token_attributions.length > 0 
            ? renderTokenHeatmap(detection.token_attributions) 
            : <p className="text-gray-500">No tokens available.</p>}
        </div>
      </div>
    </div>
  );
}
