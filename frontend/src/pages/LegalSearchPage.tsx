import { useState } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

export default function LegalSearchPage() {
  const [query, setQuery] = useState('');
  const [numCases, setNumCases] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<any>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a query description.');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const res = await axios.post(`${API_URL}/legal/query`, {
        incident_description: query,
        top_k: numCases
      });
      setResults(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to query legal service');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <header>
        <h1 className="text-3xl font-bold">📖 Medico-Legal Precedents Search</h1>
        <p className="text-gray-400 mt-2">Direct search interface to query Indian consumer court and Supreme court precedents on medical negligence.</p>
      </header>

      <form onSubmit={handleSearch} className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Enter Clinical Incident Description</label>
          <input
            type="text"
            className="w-full bg-[#0f111a] border border-[#30363d] rounded-lg p-3 text-[#e6edf3] focus:border-[#1f6feb] focus:outline-none"
            placeholder="e.g. sponge left in abdomen after surgery..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Number of cases to retrieve: {numCases}</label>
          <input
            type="range"
            min="3"
            max="10"
            value={numCases}
            onChange={(e) => setNumCases(parseInt(e.target.value))}
            className="w-full accent-[#1f6feb]"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-gray-700 text-white font-bold py-2 px-6 rounded-lg transition-colors"
        >
          {loading ? 'Searching...' : 'Search Legal Database'}
        </button>

        {error && <div className="p-4 bg-red-900/30 border border-red-500/50 text-red-400 rounded-lg">{error}</div>}
      </form>

      {results && (
        <div className="space-y-6">
          <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl space-y-4">
            <h2 className="text-xl font-semibold border-b border-[#30363d] pb-2">Legal Advisory Summary</h2>
            <div className="p-4 bg-[#0969da]/10 border border-[#0969da]/30 rounded-lg">
              <span className="font-bold text-blue-400">Standard of Care:</span> {results.standard_of_care_summary}
            </div>
            <div className="p-4 bg-[#db6d28]/10 border border-[#db6d28]/30 rounded-lg">
              <span className="font-bold text-orange-400">Liability Assessment:</span> {results.liability_assessment}
            </div>
          </div>

          <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl space-y-6">
            <h2 className="text-xl font-semibold">Relevant Case Law Precedents</h2>
            <div className="grid gap-4">
              {results.citations?.map((caseItem: any, idx: number) => (
                <details key={idx} className="group bg-[#0f111a] border border-[#30363d] rounded-lg [&_summary::-webkit-details-marker]:hidden">
                  <summary className="flex cursor-pointer items-center justify-between p-4 font-semibold text-gray-200">
                    <span>Case {idx + 1}: {caseItem.case_name} ({caseItem.year})</span>
                    <span className="transition group-open:rotate-180">
                      <svg fill="none" height="24" shapeRendering="geometricPrecision" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" viewBox="0 0 24 24" width="24"><path d="M6 9l6 6 6-6"></path></svg>
                    </span>
                  </summary>
                  <div className="p-4 pt-0 text-gray-400 border-t border-[#30363d] mt-2 space-y-2">
                    <p><strong className="text-gray-300">Court:</strong> {caseItem.court}</p>
                    <p><strong className="text-gray-300">Relevance details:</strong> {caseItem.relevance}</p>
                  </div>
                </details>
              ))}
            </div>
          </div>

          {results.statutory_provisions?.length > 0 && (
            <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl space-y-4">
              <h2 className="text-xl font-semibold">Statutory Provisions</h2>
              <ul className="space-y-2">
                {results.statutory_provisions.map((prov: string, idx: number) => (
                  <li key={idx} className="text-gray-300 flex items-start gap-2">
                    <span>⚖️</span> {prov}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
