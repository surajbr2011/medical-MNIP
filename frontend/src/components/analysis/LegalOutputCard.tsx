import React from 'react';
import type { LegalQueryResponse } from '../../types';

interface LegalOutputCardProps {
  legal: LegalQueryResponse;
}

export default function LegalOutputCard({ legal }: LegalOutputCardProps) {
  return (
    <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
      <h2 className="text-xl font-semibold mb-4">Legal Intelligence Output</h2>
      
      <p className="mb-2">
        <span className="font-bold">Standard of Care Summary:</span> {legal.standard_of_care_summary}
      </p>
      <p className="mb-6">
        <span className="font-bold">Liability Assessment:</span> {legal.liability_assessment}
      </p>
      
      <h3 className="font-semibold mb-4">Relevant Legal Precedents</h3>
      <div className="space-y-4">
        {legal.citations?.map((cit, idx) => (
          <div key={idx} className="border-b border-[#30363d] pb-4 last:border-0 last:pb-0">
            <p className="font-bold">📖 {cit.case_name} ({cit.year})</p>
            <p className="text-sm text-gray-400 mt-1">
              <span className="italic">Court:</span> {cit.court} | <span className="italic">Relevance:</span> {cit.relevance}
            </p>
          </div>
        ))}
        {(!legal.citations || legal.citations.length === 0) && (
          <p className="text-gray-500">No citations found.</p>
        )}
      </div>
    </div>
  );
}
