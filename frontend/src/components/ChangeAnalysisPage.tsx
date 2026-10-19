import React from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar,
} from 'recharts';
import { TrendingUp, TrendingDown, Building2, Trees, Waves, Layers, AlertTriangle } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useRunChangeAnalysis } from '../api/change';

const CHANGE_TYPES = [
  { key: 'all', label: 'All Changes', color: '#00d2ff' },
  { key: 'construction', label: 'Construction', color: '#f59e0b' },
  { key: 'vegetation', label: 'Vegetation', color: '#22c55e' },
  { key: 'water', label: 'Water Body', color: '#3b82f6' },
  { key: 'agriculture', label: 'Agriculture', color: '#84cc16' },
  { key: 'land_use', label: 'Land Use', color: '#a855f7' },
];

// Mock temporal data — replace with real API data when /api/change/timeline endpoint is added
const MOCK_TIMELINE_DATA = [
  { year: '2019', construction: 12, vegetation: 45, water: 8, soil: 5 },
  { year: '2020', construction: 18, vegetation: 40, water: 12, soil: 7 },
  { year: '2021', construction: 28, vegetation: 33, water: 10, soil: 9 },
  { year: '2022', construction: 38, vegetation: 26, water: 7, soil: 11 },
  { year: '2023', construction: 47, vegetation: 20, water: 9, soil: 8 },
  { year: '2024', construction: 54, vegetation: 15, water: 11, soil: 6 },
];

const MOCK_STATS = [
  { label: 'Built-up Area', value: '+42.3%', icon: Building2, color: '#f59e0b', trend: 'up' },
  { label: 'Vegetation Cover', value: '-28.7%', icon: Trees, color: '#22c55e', trend: 'down' },
  { label: 'Water Body', value: '-12.8%', icon: Waves, color: '#3b82f6', trend: 'down' },
  { label: 'Agricultural Land', value: '-9.6%', icon: Layers, color: '#84cc16', trend: 'down' },
];

export const ChangeAnalysisPage: React.FC = () => {
  const { aoi, setAOI, changeEvents } = useAppStore();
  const [activeChangeType, setActiveChangeType] = React.useState('all');
  const [dateFrom, setDateFrom] = React.useState('2019-01-01');
  const [dateTo, setDateTo] = React.useState('2024-12-31');
  const [cloudCover, setCloudCover] = React.useState('20');
  const [sensor, setSensor] = React.useState('Sentinel-2 L2A');
  const [aoiInput, setAoiInput] = React.useState({ minLon: '68.0', minLat: '8.0', maxLon: '97.0', maxLat: '37.0' });

  const runAnalysis = useRunChangeAnalysis();

  const handleRunAnalysis = () => {
    const bbox = [
      parseFloat(aoiInput.minLon),
      parseFloat(aoiInput.minLat),
      parseFloat(aoiInput.maxLon),
      parseFloat(aoiInput.maxLat),
    ];
    setAOI({ minLon: bbox[0], minLat: bbox[1], maxLon: bbox[2], maxLat: bbox[3] });
    runAnalysis.mutate({ startDate: dateFrom, endDate: dateTo, bbox });
  };

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Left Control Panel */}
      <aside className="w-[340px] bg-[#0d1527] border-r border-[#1e3a5f] flex flex-col shrink-0 overflow-y-auto">
        <div className="p-4 border-b border-[#1e3a5f]">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-4 h-4 text-[#00d2ff]" />
            <span className="text-xs font-mono font-bold text-[#00d2ff] uppercase tracking-wider">
              Change Analysis Controls
            </span>
          </div>

          {/* AOI Input */}
          <div className="mb-4">
            <label className="text-[10px] text-gray-500 font-mono uppercase block mb-2">Area of Interest (WGS84)</label>
            <div className="grid grid-cols-2 gap-2">
              {(['minLon', 'minLat', 'maxLon', 'maxLat'] as const).map((key) => (
                <div key={key}>
                  <label className="text-[9px] text-gray-600 font-mono uppercase block mb-0.5">{key}</label>
                  <input
                    type="number"
                    step="0.1"
                    value={aoiInput[key]}
                    onChange={(e) => setAoiInput((prev) => ({ ...prev, [key]: e.target.value }))}
                    className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1 text-[11px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff]"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Time Period */}
          <div className="mb-4">
            <label className="text-[10px] text-gray-500 font-mono uppercase block mb-2">Analysis Time Period</label>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[9px] text-gray-600 font-mono block mb-0.5">From</label>
                <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1 text-[11px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff] [color-scheme:dark]" />
              </div>
              <div>
                <label className="text-[9px] text-gray-600 font-mono block mb-0.5">To</label>
                <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)}
                  className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1 text-[11px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff] [color-scheme:dark]" />
              </div>
            </div>
          </div>

          {/* Change Type Filter */}
          <div className="mb-4">
            <label className="text-[10px] text-gray-500 font-mono uppercase block mb-2">Change Type</label>
            <div className="flex flex-wrap gap-1.5">
              {CHANGE_TYPES.map((ct) => (
                <button
                  key={ct.key}
                  onClick={() => setActiveChangeType(ct.key)}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono font-semibold border transition-colors ${
                    activeChangeType === ct.key
                      ? 'text-[#060a14] border-transparent'
                      : 'text-gray-400 border-[#1e3a5f] hover:border-gray-500'
                  }`}
                  style={activeChangeType === ct.key ? { backgroundColor: ct.color, borderColor: ct.color } : {}}
                >
                  {ct.label}
                </button>
              ))}
            </div>
          </div>

          {/* Sensor & Settings */}
          <div className="grid grid-cols-2 gap-2 mb-4">
            <div>
              <label className="text-[10px] text-gray-500 font-mono uppercase block mb-1">Sensor</label>
              <select value={sensor} onChange={(e) => setSensor(e.target.value)}
                className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1.5 text-[10px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff]">
                <option>Sentinel-2 L2A</option>
                <option>Sentinel-1 SAR</option>
                <option>Landsat Collection 2</option>
                <option>Multi-sensor Fusion</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 font-mono uppercase block mb-1">Max Cloud %</label>
              <input type="number" min="0" max="100" value={cloudCover} onChange={(e) => setCloudCover(e.target.value)}
                className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1.5 text-[10px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff]" />
            </div>
          </div>

          {/* Run Analysis Button */}
          <button
            onClick={handleRunAnalysis}
            disabled={runAnalysis.isPending}
            className="w-full py-2.5 bg-[#00d2ff] hover:bg-[#00b8e6] text-[#060a14] rounded-lg font-bold text-sm font-mono transition-colors disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {runAnalysis.isPending ? (
              <>
                <span className="w-4 h-4 border-2 border-[#060a14] border-t-transparent rounded-full animate-spin" />
                Running Analysis...
              </>
            ) : (
              'Run Change Analysis'
            )}
          </button>

          {runAnalysis.isSuccess && (
            <div className="mt-2 text-[11px] text-emerald-400 font-mono text-center">
              ✓ Analysis complete. {changeEvents.length} change events detected.
            </div>
          )}
          {runAnalysis.isError && (
            <div className="mt-2 text-[11px] text-red-400 font-mono text-center">
              ✗ Analysis failed. Check backend connection.
            </div>
          )}
        </div>

        {/* Change Statistics Cards */}
        <div className="p-4">
          <div className="text-[10px] text-gray-500 font-mono uppercase mb-3">Key Change Indicators</div>
          <div className="space-y-2">
            {MOCK_STATS.map((stat) => (
              <div key={stat.label} className="flex items-center justify-between p-2 bg-[#060a14] border border-[#1e3a5f] rounded-lg">
                <div className="flex items-center gap-2">
                  <stat.icon className="w-3.5 h-3.5" style={{ color: stat.color }} />
                  <span className="text-[11px] font-mono text-gray-400">{stat.label}</span>
                </div>
                <div className="flex items-center gap-1">
                  {stat.trend === 'up' ? (
                    <TrendingUp className="w-3 h-3 text-amber-400" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-red-400" />
                  )}
                  <span className={`text-xs font-bold font-mono ${stat.trend === 'up' ? 'text-amber-400' : 'text-red-400'}`}>
                    {stat.value}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Content: Timeline Chart */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Chart Area */}
        <div className="flex-1 p-6 overflow-y-auto">
          {/* KPI Cards Row */}
          <div className="grid grid-cols-4 gap-3 mb-6">
            {MOCK_STATS.map((stat) => (
              <div key={stat.label} className="bg-[#0d1527] border border-[#1e3a5f] rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <stat.icon className="w-4 h-4" style={{ color: stat.color }} />
                  {stat.trend === 'up' ? (
                    <TrendingUp className="w-3 h-3 text-amber-400" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-red-400" />
                  )}
                </div>
                <div className={`text-xl font-bold font-mono mb-1 ${stat.trend === 'up' ? 'text-amber-400' : 'text-red-400'}`}>
                  {stat.value}
                </div>
                <div className="text-[11px] text-gray-500 font-mono">{stat.label}</div>
              </div>
            ))}
          </div>

          {/* Change Timeline Chart */}
          <div className="bg-[#0d1527] border border-[#1e3a5f] rounded-xl p-4 mb-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-sm font-bold font-mono text-white">Multi-Temporal Change Timeline</div>
                <div className="text-[11px] text-gray-500 font-mono">Land cover area change (ha) 2019–2024</div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={MOCK_TIMELINE_DATA} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorConstruction" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorVegetation" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorWater" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorSoil" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#d97706" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#d97706" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="year" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e3a5f' }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e3a5f' }} unit=" ha" />
                <Tooltip
                  contentStyle={{ background: '#0d1527', border: '1px solid #1e3a5f', borderRadius: 8, fontFamily: 'monospace', fontSize: 12 }}
                  labelStyle={{ color: '#00d2ff' }}
                  itemStyle={{ color: '#f1f5f9' }}
                />
                <Legend wrapperStyle={{ fontFamily: 'monospace', fontSize: 11 }} />
                <Area type="monotone" dataKey="construction" stroke="#f59e0b" fill="url(#colorConstruction)" strokeWidth={2} name="Construction" />
                <Area type="monotone" dataKey="vegetation" stroke="#22c55e" fill="url(#colorVegetation)" strokeWidth={2} name="Vegetation" />
                <Area type="monotone" dataKey="water" stroke="#3b82f6" fill="url(#colorWater)" strokeWidth={2} name="Water Body" />
                <Area type="monotone" dataKey="soil" stroke="#d97706" fill="url(#colorSoil)" strokeWidth={2} name="Soil/Exposed" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Land Cover Bar Chart */}
          <div className="bg-[#0d1527] border border-[#1e3a5f] rounded-xl p-4">
            <div className="text-sm font-bold font-mono text-white mb-1">Land Cover Classification</div>
            <div className="text-[11px] text-gray-500 font-mono mb-4">Comparative area distribution (ha)</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={[
                { category: 'Built-up', before: 12, after: 54 },
                { category: 'Vegetation', before: 45, after: 15 },
                { category: 'Water', before: 8, after: 11 },
                { category: 'Agriculture', before: 25, after: 15 },
                { category: 'Bare Land', before: 10, after: 5 },
              ]} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="category" tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e3a5f' }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e3a5f' }} />
                <Tooltip contentStyle={{ background: '#0d1527', border: '1px solid #1e3a5f', borderRadius: 8, fontFamily: 'monospace', fontSize: 11 }} />
                <Legend wrapperStyle={{ fontFamily: 'monospace', fontSize: 10 }} />
                <Bar dataKey="before" fill="#1e3a5f" name="T1 (Baseline)" radius={[3, 3, 0, 0]} />
                <Bar dataKey="after" fill="#00d2ff" name="T2 (Current)" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
