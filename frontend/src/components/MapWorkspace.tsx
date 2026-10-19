import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Rectangle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useAppStore } from '../store/useAppStore';

// Custom Map View Updater
function MapUpdater({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  React.useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

// Marker Icons
const defaultIcon = L.divIcon({
  className: 'custom-icon',
  html: `<div style="background-color: #00d2ff; width: 14px; height: 14px; border-radius: 50%; border: 2px solid #ffffff; box-shadow: 0 0 8px #00d2ff;"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

const selectedIcon = L.divIcon({
  className: 'custom-selected-icon',
  html: `<div style="background-color: #38bdf8; width: 20px; height: 20px; border-radius: 50%; border: 3px solid #ffffff; box-shadow: 0 0 16px #38bdf8; animation: pulse 1.5s infinite;"></div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});

const changeIcon = L.divIcon({
  className: 'custom-change-icon',
  html: `<div style="background-color: #ef4444; width: 16px; height: 16px; border-radius: 50%; border: 2px solid #ffffff; box-shadow: 0 0 12px #ef4444;"></div>`,
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

export const MapWorkspace: React.FC = () => {
  const { results, selectedResult, setSelectedResult, aoi, changeEvents, selectedEvent, setSelectedEvent } = useAppStore();

  const mapCenter: [number, number] = React.useMemo(() => {
    if (selectedResult && selectedResult.center_lat && selectedResult.center_lon) {
      return [selectedResult.center_lat, selectedResult.center_lon];
    }
    if (selectedEvent && selectedEvent.location && selectedEvent.location.length === 4) {
      return [(selectedEvent.location[1] + selectedEvent.location[3]) / 2, (selectedEvent.location[0] + selectedEvent.location[2]) / 2];
    }
    if (results.length > 0 && results[0].center_lat && results[0].center_lon) {
      return [results[0].center_lat, results[0].center_lon];
    }
    return [20.5937, 78.9629]; // Default: India centroid
  }, [selectedResult, selectedEvent, results]);

  return (
    <div className="relative w-full h-full bg-[#060a14]">
      <MapContainer
        center={mapCenter}
        zoom={selectedResult ? 11 : 5}
        style={{ height: '100%', width: '100%' }}
        attributionControl={false}
      >
        <MapUpdater center={mapCenter} zoom={selectedResult ? 11 : 5} />
        {/* CartoDB Dark Matter Basemap */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          maxZoom={19}
        />

        {/* Drawn AOI Rectangle if present */}
        {aoi && (
          <Rectangle
            bounds={[
              [aoi.minLat, aoi.minLon],
              [aoi.maxLat, aoi.maxLon],
            ]}
            pathOptions={{ color: '#00d2ff', weight: 2, fillOpacity: 0.15 }}
          />
        )}

        {/* Search Results Markers */}
        {results.map((res) => {
          if (!res.center_lat || !res.center_lon) return null;
          const isSelected = selectedResult?.tile_id === res.tile_id;
          return (
            <Marker
              key={res.tile_id}
              position={[res.center_lat, res.center_lon]}
              icon={isSelected ? selectedIcon : defaultIcon}
              eventHandlers={{
                click: () => setSelectedResult(res),
              }}
            >
              <Popup className="custom-popup">
                <div className="p-2 bg-[#0d1527] text-white rounded text-xs font-mono">
                  <div className="font-bold text-[#38bdf8]">{res.source_name || res.scene_id}</div>
                  <div className="text-gray-400">Date: {res.acquisition_date}</div>
                  <div className="text-emerald-400 font-semibold">Match: {(res.similarity * 100).toFixed(1)}%</div>
                </div>
              </Popup>
            </Marker>
          );
        })}

        {/* Change Event Markers */}
        {changeEvents.map((ev) => {
          if (!ev.location || ev.location.length !== 4) return null;
          const lat = (ev.location[1] + ev.location[3]) / 2;
          const lon = (ev.location[0] + ev.location[2]) / 2;
          return (
            <Marker
              key={ev.id}
              position={[lat, lon]}
              icon={changeIcon}
              eventHandlers={{
                click: () => setSelectedEvent(ev),
              }}
            >
              <Popup>
                <div className="p-2 bg-[#0d1527] text-white rounded text-xs font-mono">
                  <div className="font-bold text-red-400">Change: {ev.change_type.toUpperCase()}</div>
                  <div className="text-yellow-400">Confidence: {(ev.confidence * 100).toFixed(0)}%</div>
                  <div className="text-gray-300">Earliest: {ev.earliest_detected_date}</div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      {/* Floating Map Legend */}
      <div className="absolute top-4 right-4 z-[1000] bg-[#0d1527]/90 border border-[#1e3a5f] rounded-lg p-3 text-xs font-mono text-gray-200 shadow-xl backdrop-blur-md">
        <div className="font-bold text-[#00d2ff] mb-2 uppercase tracking-wider">Spatial Layer Telemetry</div>
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2.5 h-2.5 rounded-full bg-[#00d2ff]"></span>
          <span>Indexed Scene Footprint</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2.5 h-2.5 rounded-full bg-[#38bdf8] ring-2 ring-white"></span>
          <span>Selected Imagery Target</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span>
          <span>Validated Change Event</span>
        </div>
      </div>
    </div>
  );
};
