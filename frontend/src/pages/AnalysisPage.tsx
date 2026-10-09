import React, { useState } from 'react';
import axios from 'axios';
import type { FullBackendReport } from '../types';
import IncidentInputForm from '../components/analysis/IncidentInputForm';
import RiskScoreCard from '../components/analysis/RiskScoreCard';
import NegligenceEngine from '../components/analysis/NegligenceEngine';
import LegalOutputCard from '../components/analysis/LegalOutputCard';

const API_URL = 'http://localhost:8000/api/v1';

export default function AnalysisPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState<FullBackendReport | null>(null);

  const handleAnalyze = async (clinicalNote: string) => {
    if (!clinicalNote.trim()) {
      setError('Please enter a clinical note to analyze.');
      return;
    }

    setLoading(true);
    setError('');
    setReport(null);

    try {
      // 1. Mock FHIR bundle for ingestion
      const fhirBundle = {
        resourceType: 'Bundle',
        type: 'transaction',
        entry: [
          { resource: { resourceType: 'Patient', id: 'pat-temp', gender: 'unknown' } },
          { resource: { resourceType: 'Encounter', id: 'enc-temp', status: 'finished', subject: { reference: 'Patient/pat-temp' } } },
          {
            resource: {
              resourceType: 'DocumentReference',
              id: 'doc-temp',
              status: 'current',
              subject: { reference: 'Patient/pat-temp' },
              context: [{ reference: 'Encounter/enc-temp' }],
              content: [{ attachment: { title: clinicalNote } }]
            }
          }
        ]
      };

      const ingestPayload = { fhir_bundle: fhirBundle, source_hospital_id: 'HOSP-IND-99' };
      
      const ingestRes = await axios.post(`${API_URL}/ingest/fhir`, ingestPayload);
      const episodeId = ingestRes.data.episode_id;

      // 2. Fetch report
      const reportRes = await axios.get(`${API_URL}/episodes/${episodeId}/report`);
      setReport(reportRes.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Pipeline execution error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold">⚕️ Clinical Incident Analysis</h1>
        <p className="text-gray-400 mt-2">Input clinical notes and optional FHIR structured data to assess negligence risk and get legal citations.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <IncidentInputForm 
          onAnalyze={handleAnalyze} 
          loading={loading} 
          error={error} 
        />

        {report?.risk ? (
          <RiskScoreCard risk={report.risk} />
        ) : report ? (
          <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl flex items-center justify-center">
            <p className="text-gray-400">No risk score details available.</p>
          </div>
        ) : null}
      </div>

      {report && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {report.detection ? (
            <NegligenceEngine detection={report.detection} />
          ) : (
            <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
              <p className="text-gray-400">No detection results available.</p>
            </div>
          )}
          
          {report.legal ? (
            <div className="h-fit">
              <LegalOutputCard legal={report.legal} />
            </div>
          ) : (
            <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl">
              <p className="text-gray-400">No legal data available.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
