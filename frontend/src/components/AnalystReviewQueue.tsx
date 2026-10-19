import React from 'react';
import { CheckCircle2, XCircle, RotateCcw, AlertTriangle, ShieldCheck } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useReviewQueue, useSubmitDecision } from '../api/change';

export const AnalystReviewQueue: React.FC = () => {
  const { changeEvents, setChangeEvents, selectedEvent, setSelectedEvent } = useAppStore();
  const { data: queueData, isLoading, refetch } = useReviewQueue();
  const submitDecision = useSubmitDecision();

  React.useEffect(() => {
    if (queueData && queueData.length > 0) {
      setChangeEvents(queueData);
      if (!selectedEvent) {
        setSelectedEvent(queueData[0]);
      }
    }
  }, [queueData]);

  const handleDecision = (eventId: string, label: 'confirm' | 'reject') => {
    submitDecision.mutate(
      {
        eventId,
        analystLabel: label,
        analystId: 'analyst_lead',
        note: `Verified through analyst portal: ${label.toUpperCase()}`,
      },
      {
        onSuccess: () => {
          refetch();
        },
      }
    );
  };

  if (isLoading) {
    return (
      <div className="p-8 text-center font-mono text-gray-400 text-sm">
        <span className="w-5 h-5 border-2 border-[#00d2ff] border-t-transparent rounded-full animate-spin inline-block mr-2"></span>
        Streaming pending analyst review candidates...
      </div>
    );
  }

  const events = changeEvents.length > 0 ? changeEvents : (queueData || []);

  if (events.length === 0) {
    return (
      <div className="p-8 text-center font-mono text-gray-400 text-xs flex flex-col items-center gap-2">
        <ShieldCheck className="w-8 h-8 text-emerald-400" />
        <span className="font-semibold text-gray-300">All Change Candidates Verified</span>
        <span className="text-gray-500">False-alarm suppression active. No unreviewed anomalies in queue.</span>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-xs">
      <div className="flex items-center justify-between text-gray-400 font-bold border-b border-[#1e3a5f] pb-2">
        <span>CANDIDATES AWAITING VERIFICATION ({events.length})</span>
        <button
          onClick={() => refetch()}
          className="text-[#00d2ff] hover:underline flex items-center gap-1 text-[11px]"
        >
          <RotateCcw className="w-3 h-3" /> Refresh
        </button>
      </div>

      {events.map((ev) => {
        const isSelected = selectedEvent?.id === ev.id;
        return (
          <div
            key={ev.id}
            onClick={() => setSelectedEvent(ev)}
            className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
              isSelected
                ? 'bg-[#111d35] border-[#00d2ff] shadow-[0_0_15px_rgba(0,210,255,0.15)]'
                : 'bg-[#0d1527] border-[#1e3a5f] hover:border-gray-500'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-red-500/20 text-red-300 border border-red-500/40">
                {ev.change_type.replace('_', ' ')}
              </span>
              <span className="text-emerald-400 font-bold">
                {(ev.confidence * 100).toFixed(0)}% Conf
              </span>
            </div>

            {/* Before / After Previews */}
            <div className="grid grid-cols-2 gap-2 my-2 bg-[#060a14] p-1.5 rounded-lg border border-[#1e3a5f]">
              <div className="relative">
                <span className="absolute bottom-1 left-1 bg-black/75 px-1 py-0.5 rounded text-[9px] text-gray-300">
                  T1: {ev.before_date || 'Baseline'}
                </span>
                {ev.before_preview ? (
                  <img src={ev.before_preview} alt="T1" className="w-full h-24 object-cover rounded" />
                ) : (
                  <div className="w-full h-24 bg-gray-900 rounded flex items-center justify-center text-gray-600">No T1</div>
                )}
              </div>

              <div className="relative">
                <span className="absolute bottom-1 left-1 bg-black/75 px-1 py-0.5 rounded text-[9px] text-gray-300">
                  T2: {ev.after_date || 'Evaluation'}
                </span>
                {ev.after_preview ? (
                  <img src={ev.after_preview} alt="T2" className="w-full h-24 object-cover rounded" />
                ) : (
                  <div className="w-full h-24 bg-gray-900 rounded flex items-center justify-center text-gray-600">No T2</div>
                )}
              </div>
            </div>

            <div className="text-[11px] text-gray-400 space-y-0.5 my-2">
              <div>Sensor: <span className="text-gray-200">{ev.sensor || 'Sentinel-2 L2A'}</span></div>
              <div>Earliest Detected: <span className="text-[#a78bfa] font-semibold">{ev.earliest_detected_date}</span></div>
              <div>Changed Footprint: <span className="text-amber-400 font-semibold">{ev.changed_area_pct}%</span></div>
            </div>

            {/* Decision Actions */}
            <div className="flex gap-2 mt-3 pt-2 border-t border-[#1e3a5f]/80">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDecision(ev.id, 'confirm');
                }}
                className="flex-1 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 border border-emerald-500/50 rounded-md font-bold text-xs flex items-center justify-center gap-1.5 transition-colors"
              >
                <CheckCircle2 className="w-3.5 h-3.5" /> Confirm
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDecision(ev.id, 'reject');
                }}
                className="flex-1 py-1.5 bg-rose-600/20 hover:bg-rose-600/40 text-rose-300 border border-rose-500/50 rounded-md font-bold text-xs flex items-center justify-center gap-1.5 transition-colors"
              >
                <XCircle className="w-3.5 h-3.5" /> Reject
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
};
