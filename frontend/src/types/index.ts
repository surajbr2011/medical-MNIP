export interface FHIRIngestion {
  patient_id: string;
  encounter_id: string;
  note_count: number;
  medication_count: number;
  procedure_count: number;
  observation_count: number;
}

export interface TokenAttribution {
  token: string;
  attribution: number;
}

export interface NegligenceResult {
  negligent: boolean;
  confidence: number;
  categories: Record<string, number>;
  token_attributions: TokenAttribution[];
}

export interface RiskScoreResponse {
  risk_score: number;
  risk_level: string;
  shap_values: Record<string, number>;
  narrative: string;
  top_drivers: string[];
}

export interface Citation {
  case_name: string;
  year: number;
  court: string;
  relevance: string;
}

export interface LegalQueryResponse {
  citations: Citation[];
  statutory_provisions: string[];
  standard_of_care_summary: string;
  liability_assessment: string;
}

export interface FullBackendReport {
  episode_id: string;
  ingestion: FHIRIngestion;
  detection?: NegligenceResult;
  risk?: RiskScoreResponse;
  legal?: LegalQueryResponse;
}

export interface LiveAnalyticsData {
  kpis: {
    total_incidents: number;
    negligence_rate: number;
    critical_alerts: number;
    latency: number;
  };
  risk_values: number[];
  cat_counts: number[];
  time_series: number[];
  top_drivers?: Array<{ name: string; value: number }>;
}
