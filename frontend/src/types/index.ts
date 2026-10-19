import React from 'react';

export interface SearchResult {
  tile_id: string;
  scene_id: string;
  source_id: string;
  source_name: string;
  acquisition_date: string;
  sensor: string;
  similarity: number;
  base_similarity: number;
  cloud_cover: number;
  center_lat: number;
  center_lon: number;
  bounds: [number, number, number, number];
  preview_path: string | null;
  description: string;
  crs: string;
  rank: number;
}

export interface ChangeEvent {
  id: string;
  before_tile_id: string;
  after_tile_id: string;
  change_type: string;
  confidence: number;
  earliest_detected_date: string;
  mask_path: string | null;
  mask_preview_path: string | null;
  changed_area_pct: number;
  status: string;
  before_date: string;
  after_date: string;
  sensor: string;
  location: [number, number, number, number];
  before_preview: string | null;
  after_preview: string | null;
  mask_preview: string | null;
  ndvi_delta?: number;
  ndwi_delta?: number;
  ndbi_delta?: number;
}

export interface AOI {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
}

export interface Scene {
  scene_id: string;
  source_id: string;
  source_name: string;
  acquisition_date: string;
  sensor: string;
  cloud_cover: number;
  tile_count: number;
  preview_path: string | null;
  bounds: number[];
}

export interface AuditEntry {
  event_id: string;
  analyst_id: string;
  analyst_label: string;
  note: string;
  timestamp: string;
}
