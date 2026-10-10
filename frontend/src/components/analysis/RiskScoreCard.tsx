import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { RiskScoreResponse } from '../../types';

interface RiskScoreCardProps {
  risk: RiskScoreResponse;
}

export default function RiskScoreCard({ risk }: RiskScoreCardProps) {
  const isHighRisk = risk.risk_level === 'High' || risk.risk_level === 'Critical';
  const isModRisk = risk.risk_level === 'Moderate';

  const badgeClass = isHighRisk 
    ? 'bg-red-900/50 text-red-400 border border-red-500/50' 
    : isModRisk 
      ? 'bg-yellow-900/50 text-yellow-400 border border-yellow-500/50' 
      : 'bg-green-900/50 text-green-400 border border-green-500/50';

  const chartData = Object.entries(risk.shap_values || {})
    .map(([name, val]) => ({ name, value: val }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8);

  return (
    <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
      <h2 className="text-xl font-semibold mb-2">1. Clinical Risk Assessment Summary</h2>
      
      <div className="flex items-baseline space-x-4 mb-2">
        <div className="text-4xl font-bold">
          {(risk.risk_score * 100).toFixed(1)}%
        </div>
        {risk.baseline_risk !== undefined && (
          <div className="text-sm text-gray-400">
            Baseline Prior: {(risk.baseline_risk * 100).toFixed(1)}%
          </div>
        )}
      </div>
      
      <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold mb-4 ${badgeClass}`}>
        Risk Severity Level: {risk.risk_level}
      </div>
      
      <div className="bg-[#0f111a] p-4 rounded-lg border border-[#30363d] text-gray-300 text-sm mb-6 leading-relaxed">
        <span className="font-bold text-white">Clinical Narrative:</span> {risk.narrative}
      </div>
      
      <h3 className="font-semibold mb-1 text-sm text-gray-200">Feature Contributions (TreeExplainer Log-Odds Space)</h3>
      <p className="text-xs text-gray-400 mb-2">Values show marginal shift in model log-odds relative to expected baseline. Not direct percentage multipliers.</p>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart layout="vertical" data={chartData} margin={{ left: 80 }}>
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={120} tick={{ fill: '#8b949e', fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} />
            <Bar dataKey="value" fill="#1f6feb" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-gray-500 mt-4 border-t border-[#30363d] pt-2">
        ⚖️ {risk.disclaimer || "Statistical risk associations based on clinical features; not proof of clinical causation or negligence."}
      </p>
    </div>
  );
}
