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
        <h2 className="text-xl font-semibold mb-2">2. Negligence Screening Engine (Bio_ClinicalBERT)</h2>
        
        {(() => {
          const status = detection.status || (detection.negligent ? "Flagged for Clinical Review" : "Unflagged (Low Suspicion)");
          const isEquivocal = status.includes("Equivocal");
          const isFlagged = detection.negligent;

          const statusStyle = isEquivocal
            ? "bg-amber-950/40 border-amber-500/50 text-amber-300"
            : isFlagged
              ? "bg-rose-950/40 border-rose-500/50 text-rose-300"
              : "bg-sky-950/40 border-sky-500/50 text-sky-300";

          return (
            <div className={`p-4 rounded-lg border mb-4 ${statusStyle}`}>
              <div className="font-semibold text-base mb-1">
                Screening Status: {status}
              </div>
              <div className="text-sm opacity-90">
                Class Confidence: {(detection.confidence * 100).toFixed(1)}% | Raw Probability: {((detection.screening_probability ?? detection.confidence) * 100).toFixed(1)}% (Threshold: {(detection.decision_threshold ?? 0.50).toFixed(2)})
              </div>
            </div>
          );
        })()}

        <p className="text-xs text-gray-400 mb-6 bg-[#0f111a] p-3 rounded border border-[#30363d]">
          ⚠️ <strong>Governance Notice:</strong> AI-assisted screening prioritization tool. Designed for clinical triage, not an autonomous legal determination of medical negligence.
        </p>
        
        <h3 className="font-semibold mb-1 text-sm text-gray-200">WHO ICPS Multi-Domain Breakdown</h3>
        <p className="text-xs text-gray-400 mb-2">Independent sigmoid probabilities per domain (non-mutually exclusive). Incidents may involve multiple interacting categories.</p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={chartData}>
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={110} tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} />
              <Bar dataKey="value" fill="#238636" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
        <h2 className="text-xl font-semibold mb-1">Explainability: Deep Learning Token Attribution</h2>
        <p className="text-xs text-gray-400 mb-4">
          Signed token attribution via <strong>Input × Gradient</strong> on Bio_ClinicalBERT token embeddings. Highlights indicate vocabulary associated with the statistical classification, not proof of clinical fault.
        </p>
        <div className="bg-[#1e1e1e] p-4 rounded-lg border border-[#333] leading-loose">
          {detection.token_attributions && detection.token_attributions.length > 0 
            ? renderTokenHeatmap(detection.token_attributions) 
            : <p className="text-gray-500">No tokens available.</p>}
        </div>
      </div>
    </div>
  );
}
