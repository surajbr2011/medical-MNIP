import React, { useState } from 'react';

interface IncidentInputFormProps {
  onAnalyze: (clinicalNote: string) => void;
  loading: boolean;
  error: string;
}

export default function IncidentInputForm({ onAnalyze, loading, error }: IncidentInputFormProps) {
  const [clinicalNote, setClinicalNote] = useState('');

  const handleSubmit = () => {
    onAnalyze(clinicalNote);
  };

  return (
    <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl h-full">
      <h2 className="text-xl font-semibold mb-4">1. Clinical Data Input</h2>
      <textarea
        className="w-full h-48 bg-[#0f111a] border border-[#30363d] rounded-lg p-4 text-[#e6edf3] focus:border-[#1f6feb] focus:outline-none mb-4"
        placeholder="Type or paste patient summary clinical note here..."
        value={clinicalNote}
        onChange={(e) => setClinicalNote(e.target.value)}
      />
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="w-full bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg transition-colors"
      >
        {loading ? 'Processing...' : 'Run MNIP Analysis'}
      </button>
      
      {error && (
        <div className="mt-4 p-4 bg-red-900/30 border border-red-500/50 text-red-400 rounded-lg">
          {error}
        </div>
      )}
    </div>
  );
}
