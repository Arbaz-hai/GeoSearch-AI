import React from 'react';
import { Calendar, Play, Layers } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const TimelineScrubber: React.FC = () => {
  const { t1Date, setT1Date, t2Date, setT2Date, maskOpacity, setMaskOpacity, selectedResult } = useAppStore();

  const mockAcquisitions = [
    { date: '2023-01-15', cloud: 4 },
    { date: '2023-04-12', cloud: 12 },
    { date: '2023-07-28', cloud: 8 },
    { date: '2023-10-10', cloud: 2 },
    { date: '2024-01-20', cloud: 18 },
    { date: '2024-05-14', cloud: 6 },
  ];

  return (
    <div className="absolute bottom-4 left-6 right-6 z-[1000] bg-[#0d1527]/95 border border-[#1e3a5f] rounded-xl p-4 shadow-2xl backdrop-blur-md font-mono text-xs">
      <div className="flex items-center justify-between mb-3 border-b border-[#1e3a5f] pb-2">
        <div className="flex items-center gap-2 text-[#00d2ff] font-bold">
          <Calendar className="w-4 h-4" />
          <span>MULTI-TEMPORAL TIMELINE & SPECTRAL DRIFT SCRUBBER</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-gray-400 text-[11px]">Change Mask Blend:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={maskOpacity}
            onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
            className="w-24 accent-[#00d2ff] cursor-pointer"
          />
          <span className="text-[#38bdf8] font-bold w-8 text-right">{(maskOpacity * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Epoch Scrub Markers */}
      <div className="relative flex items-center justify-between py-2 px-6">
        <div className="absolute left-6 right-6 top-1/2 h-0.5 bg-[#1e3a5f] -translate-y-1/2"></div>
        {mockAcquisitions.map((acq, idx) => {
          const isT1 = t1Date === acq.date;
          const isT2 = t2Date === acq.date;
          return (
            <div key={idx} className="relative z-10 flex flex-col items-center group cursor-pointer">
              <button
                type="button"
                onClick={() => {
                  if (!t1Date || (t1Date && t2Date)) {
                    setT1Date(acq.date);
                    setT2Date(null);
                  } else {
                    setT2Date(acq.date);
                  }
                }}
                className={`w-4 h-4 rounded-full border-2 transition-transform transform group-hover:scale-125 ${
                  isT1
                    ? 'bg-[#38bdf8] border-white shadow-[0_0_10px_#38bdf8]'
                    : isT2
                    ? 'bg-red-500 border-white shadow-[0_0_10px_#ef4444]'
                    : acq.cloud < 10
                    ? 'bg-emerald-500 border-[#0d1527]'
                    : 'bg-yellow-500 border-[#0d1527]'
                }`}
              />
              <span className="text-[10px] text-gray-300 mt-1 font-semibold">{acq.date}</span>
              <span className="text-[9px] text-gray-500">{acq.cloud}% cloud</span>
            </div>
          );
        })}
      </div>

      <div className="flex items-center justify-between mt-2 pt-2 border-t border-[#1e3a5f]/60 text-[11px] text-gray-400">
        <div>
          Baseline Epoch (T1): <strong className="text-[#38bdf8]">{t1Date || 'Select from scrub bar'}</strong>
        </div>
        <div>
          Evaluation Epoch (T2): <strong className="text-red-400">{t2Date || 'Select from scrub bar'}</strong>
        </div>
        <div className="text-gray-500 italic">
          Prithvi spatiotemporal cosine divergence evaluated at 10m GSD
        </div>
      </div>
    </div>
  );
};
