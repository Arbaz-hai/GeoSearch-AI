import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Satellite, Eye, Layers, Activity, RefreshCw } from 'lucide-react';
import { useAppStore } from './store/useAppStore';
import { MapWorkspace } from './components/MapWorkspace';
import { SemanticQueryBar } from './components/SemanticQueryBar';
import { TimelineScrubber } from './components/TimelineScrubber';
import { AnalystReviewQueue } from './components/AnalystReviewQueue';
import { ChangeAnalysisPage } from './components/ChangeAnalysisPage';
import { ClusterDiscovery } from './components/ClusterDiscovery';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 1000 * 60 * 5,
    },
  },
});

const MainDashboard: React.FC = () => {
  const { activeTab, setActiveTab, results, selectedResult, setSelectedResult } = useAppStore();

  return (
    <div className="h-screen w-screen flex flex-col bg-[#060a14] text-gray-100 overflow-hidden font-sans">
      {/* Top Header */}
      <header className="h-14 bg-[#0d1527] border-b border-[#1e3a5f] px-6 flex items-center justify-between shrink-0 shadow-lg z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[#00d2ff]/10 border border-[#00d2ff]/30 text-[#00d2ff]">
            <Satellite className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold text-base tracking-tight text-white">GeoSearch AI</h1>
              <span className="text-[10px] font-mono uppercase bg-[#1e3a5f] text-[#38bdf8] px-2 py-0.5 rounded font-bold">
                Triple-Encoder v2.1
              </span>
            </div>
            <p className="text-[11px] text-gray-400 font-mono">Ministry of Defence (MoD) · Indian Army (DGIS)</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 font-mono text-xs">
          <button
            onClick={() => setActiveTab('search')}
            className={`px-3.5 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-colors ${
              activeTab === 'search'
                ? 'bg-[#00d2ff]/15 text-[#00d2ff] border border-[#00d2ff]/40'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" /> Semantic Discovery
          </button>
          <button
            onClick={() => setActiveTab('change')}
            className={`px-3.5 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-colors ${
              activeTab === 'change'
                ? 'bg-[#00d2ff]/15 text-[#00d2ff] border border-[#00d2ff]/40'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> Change Analysis & Review
          </button>
          <button
            onClick={() => setActiveTab('clusters')}
            className={`px-3.5 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-colors ${
              activeTab === 'clusters'
                ? 'bg-[#00d2ff]/15 text-[#00d2ff] border border-[#00d2ff]/40'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Activity className="w-3.5 h-3.5" /> Cluster Discovery
          </button>
        </nav>

        {/* Status Indicators */}
        <div className="flex items-center gap-3 font-mono text-xs text-gray-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#22c55e]"></span>
            <span className="text-[11px]">Air-Gap Sovereign Mode</span>
          </div>
          <div className="h-4 w-px bg-[#1e3a5f]"></div>
          <div className="text-[11px] text-[#38bdf8]">FAISS Triple-Index Active</div>
        </div>
      </header>

      {/* Main Workspace Body */}
      {activeTab === 'change' ? (
        <div className="flex-1 flex overflow-hidden">
          <ChangeAnalysisPage />
          <div className="w-[380px] bg-[#0d1527] border-l border-[#1e3a5f] flex flex-col shrink-0">
            <AnalystReviewQueue />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex overflow-hidden">
          {/* Left Sidepanel */}
          <aside className="w-[440px] bg-[#0d1527] border-r border-[#1e3a5f] flex flex-col shrink-0">
            {activeTab === 'search' && (
              <>
                <SemanticQueryBar />
                {/* Search Results List */}
                <div className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-xs">
                  <div className="text-gray-400 font-bold uppercase border-b border-[#1e3a5f] pb-2 text-[11px]">
                    Ranked Imagery Results ({results.length})
                  </div>
                  {results.length === 0 ? (
                    <div className="p-8 text-center text-gray-500 text-xs">
                      Enter a natural language query or pick a suggestion above.
                    </div>
                  ) : (
                    results.map((res) => {
                      const isSelected = selectedResult?.tile_id === res.tile_id;
                      return (
                        <div
                          key={res.tile_id}
                          onClick={() => setSelectedResult(res)}
                          className={`p-3 rounded-lg border transition-all cursor-pointer ${
                            isSelected
                              ? 'bg-[#111d35] border-[#00d2ff] shadow-[0_0_12px_rgba(0,210,255,0.2)]'
                              : 'bg-[#060a14] border-[#1e3a5f] hover:border-gray-500'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="font-bold text-[#38bdf8] truncate max-w-[200px]">
                              {res.source_name || res.scene_id}
                            </span>
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-[#00d2ff]/10 text-[#00d2ff] border border-[#00d2ff]/30">
                              {(res.similarity * 100).toFixed(1)}% Match
                            </span>
                          </div>
                          <div className="text-[11px] text-gray-400 mb-2">
                            Acquired: <span className="text-gray-200">{res.acquisition_date}</span> · {res.sensor}
                          </div>
                          {res.preview_path && (
                            <img
                              src={res.preview_path}
                              alt="preview"
                              className="w-full h-32 object-cover rounded border border-[#1e3a5f]"
                            />
                          )}
                        </div>
                      );
                    })
                  )}
                </div>
              </>
            )}

            {activeTab === 'clusters' && <ClusterDiscovery />}
          </aside>

          {/* Right Map Viewport */}
          <main className="flex-1 relative">
            <MapWorkspace />
            <TimelineScrubber />
          </main>
        </div>
      )}
    </div>
  );
};

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MainDashboard />
    </QueryClientProvider>
  );
}
