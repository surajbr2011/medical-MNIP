import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import React from 'react';
import type { LiveAnalyticsData } from '../types';
import MetricCard from '../components/analytics/MetricCard';
import ChartWidget from '../components/analytics/ChartWidget';

const API_URL = 'http://localhost:8000/api/v1';

export default function AnalyticsPage() {
  const { data, isLoading, isError } = useQuery<LiveAnalyticsData>({
    queryKey: ['live-analytics'],
    queryFn: async () => {
      const res = await axios.get(`${API_URL}/analytics/live`);
      return res.data;
    },
    refetchInterval: 2000,
  });

  const riskLabels = ["Minimal", "Moderate", "High", "Critical"];
  const COLORS = ["#2da44e", "#0969da", "#db6d28", "#cf222e"];
  const pieData = data?.risk_values.map((val, index) => ({ name: riskLabels[index], value: val })) || [];

  const WHO_ICPS_CATEGORIES = [
    "Clinical Process / Procedure",
    "Medication / IV Fluids",
    "Healthcare-associated Infection",
    "Medical Device / Equipment",
    "Patient Accidents / Falls",
    "Infrastructure / Environment",
    "Resources / Organizational",
    "Documentation"
  ];
  const barData = data?.cat_counts.map((val, idx) => ({ name: WHO_ICPS_CATEGORIES[idx], value: val }))
    .sort((a, b) => a.value - b.value) || [];

  const weeks = Array.from({ length: 12 }, (_, i) => `Wk ${i - 11}`);
  const lineData = data?.time_series.map((val, i) => ({ name: weeks[i], value: val })) || [];

  const drivers = [
    "treatment_delay_hours",
    "vital_sign_deterioration_flag",
    "consent_documented_flag_missing",
    "high_risk_drug_flag",
    "staff_patient_ratio"
  ];
  const driverImpact = [0.24, 0.18, 0.15, 0.12, 0.09];
  const driverData = driverImpact.map((val, idx) => ({ name: drivers[idx], value: val })).reverse();

  if (isLoading) return <div className="p-8 text-xl">Loading live analytics dashboard...</div>;
  if (isError || !data) return <div className="p-8 text-xl text-red-500">Error connecting to live analytics backend.</div>;

  return (
    <div className="space-y-6">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            📊 Platform Executive Analytics
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
          </h1>
          <p className="text-gray-400 mt-2">Live real-time reporting on negligence incidents, risks, and trends.</p>
        </div>
      </header>

      <h2 className="text-xl font-semibold mt-6">Live Incidents Feed</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard title="Total Incidents Processed" value={data.kpis.total_incidents} colorClass="text-green-400" />
        <MetricCard title="Negligence Confirmed Rate" value={`${data.kpis.negligence_rate}%`} colorClass="text-red-400" />
        <MetricCard title="Critical Risk Alerts" value={data.kpis.critical_alerts} colorClass="text-red-400" />
        <MetricCard title="Avg Latency (seconds)" value={`${data.kpis.latency}s`} colorClass="text-green-400" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartWidget 
          title="Risk Level Distribution (Live)" 
          footer={
            <div className="flex justify-center gap-4 text-sm text-gray-300">
              {pieData.map((d, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i] }}></span>
                  {d.name} ({d.value})
                </div>
              ))}
            </div>
          }
        >
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={5} dataKey="value" animationDuration={1000}>
                {pieData.map((_, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} itemStyle={{ color: '#fff' }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartWidget>

        <ChartWidget title="Flagged Incidents by Domain (Live)" heightClass="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={barData} margin={{ left: 160, right: 20 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" tick={{ fill: '#8b949e', fontSize: 11 }} width={160} />
              <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} itemStyle={{ color: '#fff' }} />
              <Bar dataKey="value" fill="#8957e5" radius={[0, 4, 4, 0]} animationDuration={1000} />
            </BarChart>
          </ResponsiveContainer>
        </ChartWidget>

        <ChartWidget title="Incident Ingestion Trend">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lineData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 12 }} />
              <YAxis tick={{ fill: '#8b949e', fontSize: 12 }} />
              <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} itemStyle={{ color: '#fff' }} />
              <Line type="monotone" dataKey="value" stroke="#1f6feb" strokeWidth={3} dot={{ fill: '#1f6feb', r: 4 }} activeDot={{ r: 6 }} animationDuration={1000} />
            </LineChart>
          </ResponsiveContainer>
        </ChartWidget>

        <ChartWidget title="Top 5 Negligence Risk Drivers">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={driverData} margin={{ left: 180, right: 20 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" tick={{ fill: '#8b949e', fontSize: 11 }} width={180} />
              <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d' }} itemStyle={{ color: '#fff' }} />
              <Bar dataKey="value" fill="#db6d28" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartWidget>
      </div>
    </div>
  );
}
