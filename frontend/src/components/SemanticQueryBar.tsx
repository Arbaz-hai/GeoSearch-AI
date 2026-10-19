import React, { useState } from 'react';
import { Search, Satellite, MapPin, Clock, CloudSun, ChevronRight } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useTextSearch } from '../api/search';

const SAMPLE_QUERIES = [
  'Newly built structures near a river',
  'Large vehicle concentrations on open ground',
  'Flood-affected agricultural land',
  'Deforestation activity in forested area',
  'Urban expansion on green belt',
  'Construction site development',
];

const SENSORS = [
  'All Sensors',
  'Sentinel-2 L2A',
  'Sentinel-1 SAR',
  'Landsat Collection 2',
  'Bhuvan / NRSC',
];

export const SemanticQueryBar: React.FC = () => {
  const {
    query, setQuery,
    results, setResults,
    selectedSensor, setSelectedSensor,
    dateStart, setDateStart,
    dateEnd, setDateEnd,
  } = useAppStore();

  const [localQuery, setLocalQuery] = useState(query);
  const [searchEnabled, setSearchEnabled] = useState(false);

  const { data, isLoading, error } = useTextSearch(
    query,
    { sensor: selectedSensor === 'All Sensors' ? undefined : selectedSensor, date_start: dateStart || undefined, date_end: dateEnd || undefined },
    searchEnabled
  );

  React.useEffect(() => {
    if (data) {
      setResults(data);
      setSearchEnabled(false);
    }
  }, [data]);

  const handleSearch = () => {
    if (!localQuery.trim()) return;
    setQuery(localQuery);
    setSearchEnabled(true);
  };

  const handleSampleQuery = (q: string) => {
    setLocalQuery(q);
    setQuery(q);
    setSearchEnabled(true);
  };

  return (
    <div className="p-4 border-b border-[#1e3a5f] bg-[#0d1527] shrink-0">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Search className="w-4 h-4 text-[#00d2ff]" />
        <span className="text-xs font-mono font-bold text-[#00d2ff] uppercase tracking-wider">
          Semantic Intelligence Query
        </span>
      </div>

      {/* Main Search Input */}
      <div className="flex gap-2 mb-3">
        <input
          type="text"
          value={localQuery}
          onChange={(e) => setLocalQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="e.g. newly built structures near a river..."
          className="flex-1 bg-[#060a14] border border-[#1e3a5f] rounded-lg px-3 py-2 text-xs font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-[#00d2ff] focus:ring-1 focus:ring-[#00d2ff]/30 transition-colors"
        />
        <button
          onClick={handleSearch}
          disabled={isLoading}
          className="px-4 py-2 bg-[#00d2ff] hover:bg-[#00b8e6] text-[#060a14] rounded-lg font-bold text-xs font-mono transition-colors disabled:opacity-50 flex items-center gap-1.5"
        >
          {isLoading ? (
            <span className="w-3.5 h-3.5 border-2 border-[#060a14] border-t-transparent rounded-full animate-spin" />
          ) : (
            <Search className="w-3.5 h-3.5" />
          )}
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {/* Filters Row */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div>
          <label className="text-[10px] text-gray-500 font-mono uppercase block mb-1">Sensor</label>
          <select
            value={selectedSensor}
            onChange={(e) => setSelectedSensor(e.target.value)}
            className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1.5 text-[11px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff]"
          >
            {SENSORS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-gray-500 font-mono uppercase block mb-1">Date From</label>
          <input
            type="date"
            value={dateStart}
            onChange={(e) => setDateStart(e.target.value)}
            className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1.5 text-[11px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff] [color-scheme:dark]"
          />
        </div>
        <div>
          <label className="text-[10px] text-gray-500 font-mono uppercase block mb-1">Date To</label>
          <input
            type="date"
            value={dateEnd}
            onChange={(e) => setDateEnd(e.target.value)}
            className="w-full bg-[#060a14] border border-[#1e3a5f] rounded px-2 py-1.5 text-[11px] font-mono text-gray-300 focus:outline-none focus:border-[#00d2ff] [color-scheme:dark]"
          />
        </div>
      </div>

      {/* Sample Queries */}
      <div>
        <div className="text-[10px] text-gray-500 font-mono uppercase mb-1.5">Example Queries</div>
        <div className="flex flex-wrap gap-1.5">
          {SAMPLE_QUERIES.slice(0, 4).map((q) => (
            <button
              key={q}
              onClick={() => handleSampleQuery(q)}
              className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-[#1e3a5f] text-gray-400 hover:border-[#00d2ff] hover:text-[#00d2ff] transition-colors"
            >
              {q.length > 28 ? q.slice(0, 28) + '…' : q}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mt-2 text-[10px] text-red-400 font-mono bg-red-900/20 border border-red-700/30 rounded p-2">
          Search failed: {String(error)}
        </div>
      )}
    </div>
  );
};
