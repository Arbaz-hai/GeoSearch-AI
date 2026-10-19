import React from 'react';
import { Network, Crosshair, MapPin, BarChart2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useCluster } from '../api/change';
import { SearchResult } from '../types';

export const ClusterDiscovery: React.FC = () => {
  const { selectedResult, clusterNeighbors, setClusterNeighbors, setSelectedResult } = useAppStore();
  const { data: clusterData, isLoading } = useCluster(selectedResult?.tile_id ?? null);

  React.useEffect(() => {
    if (clusterData && Array.isArray((clusterData as any).neighbors)) {
      setClusterNeighbors((clusterData as any).neighbors);
    }
  }, [clusterData]);

  const neighbors = clusterNeighbors.length > 0 ? clusterNeighbors : [];

  return (
    <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-[#1e3a5f] pb-3">
        <Network className="w-4 h-4 text-[#00d2ff]" />
        <div>
          <div className="font-bold text-[#00d2ff] uppercase tracking-wider text-[11px]">HDBSCAN Site Discovery</div>
          <div className="text-gray-500 text-[10px]">Similar sites via FAISS 512-D embedding space</div>
        </div>
      </div>

      {/* Info Card */}
      <div className="bg-[#060a14] border border-[#1e3a5f] rounded-lg p-3">
        <div className="text-emerald-400 font-bold mb-1 text-[11px]">Active Learning Feedback Loop</div>
        <div className="text-gray-400 text-[10px] leading-relaxed">
          Confirm a change event to boost cluster recall by +0.05. FAISS searches the 512-D RemoteCLIP embedding space to surface structurally and semantically similar sites globally.
        </div>
      </div>

      {/* Current Selection */}
      {selectedResult ? (
        <div className="bg-[#0d1527] border border-[#00d2ff]/40 rounded-lg p-3">
          <div className="text-[10px] text-gray-500 uppercase mb-1">Selected Reference Site</div>
          <div className="font-bold text-[#38bdf8]">{selectedResult.source_name || selectedResult.scene_id}</div>
          <div className="text-gray-400 text-[10px]">
            {selectedResult.acquisition_date} · {selectedResult.sensor}
          </div>
          <div className="text-[#00d2ff] font-bold mt-1">
            {(selectedResult.similarity * 100).toFixed(1)}% semantic match
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-500 py-6 text-[11px]">
          <Crosshair className="w-6 h-6 mx-auto mb-2 text-gray-700" />
          Select a scene from Search tab to discover similar sites
        </div>
      )}

      {/* Cluster Neighbors */}
      {isLoading && (
        <div className="text-center text-gray-500 py-4">
          <span className="w-4 h-4 border-2 border-[#00d2ff] border-t-transparent rounded-full animate-spin inline-block" />
          <span className="ml-2">Searching embedding space...</span>
        </div>
      )}

      {neighbors.length > 0 && (
        <div>
          <div className="text-[10px] text-gray-500 uppercase mb-2 flex items-center gap-1">
            <BarChart2 className="w-3 h-3" />
            Similar Sites ({neighbors.length})
          </div>
          <div className="space-y-2">
            {neighbors.map((site: SearchResult, i: number) => (
              <div
                key={site.tile_id}
                onClick={() => setSelectedResult(site)}
                className="flex items-center gap-3 p-2.5 bg-[#060a14] border border-[#1e3a5f] rounded-lg hover:border-[#00d2ff]/50 cursor-pointer transition-colors"
              >
                <div className="text-gray-600 font-bold text-sm w-5 shrink-0">#{i + 1}</div>
                {site.preview_path ? (
                  <img src={site.preview_path} alt="preview" className="w-10 h-10 rounded object-cover shrink-0 border border-[#1e3a5f]" />
                ) : (
                  <div className="w-10 h-10 rounded bg-[#0d1527] border border-[#1e3a5f] flex items-center justify-center shrink-0">
                    <MapPin className="w-4 h-4 text-gray-600" />
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-[#38bdf8] font-bold truncate">{site.source_name || site.scene_id}</div>
                  <div className="text-gray-500 text-[10px]">{site.acquisition_date}</div>
                </div>
                <div className="text-[#00d2ff] font-bold text-[11px] shrink-0">
                  {(site.similarity * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!isLoading && neighbors.length === 0 && selectedResult && (
        <div className="text-center text-gray-600 py-4 text-[11px]">
          No similar sites found in current index. Ingest more scenes to expand discovery coverage.
        </div>
      )}
    </div>
  );
};
