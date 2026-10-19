import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { SearchResult, ChangeEvent, AOI } from '../types';

interface AppState {
  // Navigation
  activeTab: 'search' | 'change' | 'clusters' | 'ingest';
  setActiveTab: (tab: 'search' | 'change' | 'clusters' | 'ingest') => void;

  // Search State
  query: string;
  setQuery: (q: string) => void;
  selectedSensor: string;
  setSelectedSensor: (sensor: string) => void;
  dateStart: string;
  setDateStart: (d: string) => void;
  dateEnd: string;
  setDateEnd: (d: string) => void;
  aoi: AOI | null;
  setAOI: (aoi: AOI | null) => void;

  results: SearchResult[];
  setResults: (results: SearchResult[]) => void;
  selectedResult: SearchResult | null;
  setSelectedResult: (res: SearchResult | null) => void;

  // Change Detection State
  t1Date: string | null;
  setT1Date: (d: string | null) => void;
  t2Date: string | null;
  setT2Date: (d: string | null) => void;
  maskOpacity: number;
  setMaskOpacity: (opacity: number) => void;
  changeEvents: ChangeEvent[];
  setChangeEvents: (events: ChangeEvent[]) => void;
  selectedEvent: ChangeEvent | null;
  setSelectedEvent: (ev: ChangeEvent | null) => void;

  // Active Learning / Clusters
  clusterNeighbors: SearchResult[];
  setClusterNeighbors: (neighbors: SearchResult[]) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      activeTab: 'search',
      setActiveTab: (activeTab) => set({ activeTab }),

      query: '',
      setQuery: (query) => set({ query }),
      selectedSensor: 'All Sensors',
      setSelectedSensor: (selectedSensor) => set({ selectedSensor }),
      dateStart: '',
      setDateStart: (dateStart) => set({ dateStart }),
      dateEnd: '',
      setDateEnd: (dateEnd) => set({ dateEnd }),
      aoi: null,
      setAOI: (aoi) => set({ aoi }),

      results: [],
      setResults: (results) => set({ results }),
      selectedResult: null,
      setSelectedResult: (selectedResult) => set({ selectedResult }),

      t1Date: null,
      setT1Date: (t1Date) => set({ t1Date }),
      t2Date: null,
      setT2Date: (t2Date) => set({ t2Date }),
      maskOpacity: 0.65,
      setMaskOpacity: (maskOpacity) => set({ maskOpacity }),
      changeEvents: [],
      setChangeEvents: (changeEvents) => set({ changeEvents }),
      selectedEvent: null,
      setSelectedEvent: (selectedEvent) => set({ selectedEvent }),

      clusterNeighbors: [],
      setClusterNeighbors: (clusterNeighbors) => set({ clusterNeighbors }),
    }),
    {
      name: 'geosearch-ai-session',
    }
  )
);
