# GeoSearch AI

**Earth Observation Semantic Search & Temporal Change Detection**

---

## Technology Stack

### Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white)

### Frontend
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![React Leaflet](https://img.shields.io/badge/React_Leaflet-199900?style=flat-square&logo=leaflet&logoColor=white)
![Zustand](https://img.shields.io/badge/Zustand-443E38?style=flat-square)
![Leaflet Draw](https://img.shields.io/badge/Leaflet_Draw-199900?style=flat-square&logo=leaflet&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)
![Deck.gl](https://img.shields.io/badge/Deck.gl-000000?style=flat-square)
![TanStack Query](https://img.shields.io/badge/TanStack_Query-FF4154?style=flat-square&logo=reactquery&logoColor=white)

### Backend + AI
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4B32C3?style=flat-square)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white)
![RemoteCLIP](https://img.shields.io/badge/RemoteCLIP-00B4D8?style=flat-square)
![Prithvi-EO-2.0](https://img.shields.io/badge/Prithvi--EO--2.0-2D6A4F?style=flat-square)
![SkySense++](https://img.shields.io/badge/SkySense++-7B2FBE?style=flat-square)
![ChangeFormer](https://img.shields.io/badge/ChangeFormer-1D3557?style=flat-square)

### Search + Geo
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square)
![HDBSCAN](https://img.shields.io/badge/HDBSCAN-E040FB?style=flat-square)
![UMAP](https://img.shields.io/badge/UMAP-FF6D00?style=flat-square)
![Rasterio](https://img.shields.io/badge/Rasterio-3D9970?style=flat-square)
![GDAL/OGR](https://img.shields.io/badge/GDAL%2FOGR-5AAA46?style=flat-square)
![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?style=flat-square)
![Shapely](https://img.shields.io/badge/Shapely-2E8B57?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

### Data + Storage
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Sentinel-1](https://img.shields.io/badge/Sentinel--1-1565C0?style=flat-square)
![Sentinel-2](https://img.shields.io/badge/Sentinel--2-1976D2?style=flat-square)
![HLS](https://img.shields.io/badge/HLS-0288D1?style=flat-square)
![GeoTIFF/COG](https://img.shields.io/badge/GeoTIFF%2FCOG-5C6BC0?style=flat-square)

---

## 1. Abstract

**GeoSearch AI** is a multimodal search and analysis engine designed for intelligence analysts, researchers, and geospatial professionals. It transcends traditional metadata-based querying by enabling **Natural Language Semantic Search** and **Image-to-Image similarity mapping** across large-scale remote sensing archives.

Built with a strict **Offline-First** architecture, the system guarantees that sensitive telemetry and imagery never leave the local environment, executing foundation models (RemoteCLIP, Prithvi-EO-2.0, SkySense++, ChangeFormer) entirely on edge or isolated hardware.

---

## 2. Core Capabilities

* **Multimodal Semantic Search:** Query satellite imagery using natural language (e.g., *"Show me industrial facilities near water bodies"*).
* **Automated Change Detection:** Perform temporal analysis between historical and current scenes using ChangeFormer to identify infrastructural or environmental deviations.
* **Offline-First Execution:** Fully operational in air-gapped environments. All models are staged and executed locally.
* **Multi-Signal Ranking Algorithm:** A sophisticated scoring engine that blends Cosine Similarity from FAISS vector search with lexical keyword boosts and quality metric penalties (e.g., cloud cover) for high-precision retrieval.
* **Geospatial Clustering:** HDBSCAN + UMAP for spatial pattern discovery across large archives.
* **Automated Metadata Extraction:** Ingestion pipelines automatically generate captions and semantic tags for unlabeled imagery.

---

## 3. Application Interface

### Website Overview
![Website Overview](data/imagery/Webiste%20page%201.png)

### AI Processing Pipeline
![AI Pipeline](data/imagery/sentinel2/ai%20pipeline%20new.jpeg)

---

## 4. Architecture & Processing Pipelines

GeoSearch AI utilizes a highly modular pipeline-driven architecture.

### 4.1 System Topology

```mermaid
graph TD
    User([User / Analyst])

    subgraph Client Interface
        G[React + Vite Frontend]
        L[React Leaflet Map]
    end

    subgraph Core Services
        API[FastAPI + Uvicorn]
        Auth[HMAC Authentication]
        API --> Auth
    end

    subgraph Execution Pipelines
        IP[Ingestion Pipeline]
        SP[Search Pipeline]
        CP[Change Detection Pipeline]
    end

    subgraph Foundation Models
        RCLIP[RemoteCLIP]
        PRITHVI[Prithvi-EO-2.0]
        SKYSENSE[SkySense++]
        CF[ChangeFormer]
    end

    subgraph Search & Geo
        FAISS[(FAISS Index)]
        HDBSCAN[HDBSCAN Clustering]
        GDAL[GDAL/OGR + Rasterio]
    end

    subgraph Persistence Layer
        SQLite[(SQLite Metadata)]
        Storage[Local Image Store]
    end

    User <-->|HTTP/REST| G
    G <--> L
    G <--> API
    API --> IP
    API --> SP
    API --> CP

    IP --> RCLIP & PRITHVI & GDAL
    SP --> RCLIP & FAISS & HDBSCAN
    CP --> CF & SKYSENSE

    IP --> SQLite & Storage
    SP --> SQLite & FAISS
    CP --> Storage
```

### 4.2 Ingestion & Semantic Tagging

During the ingestion of raster data (GeoTIFF, Sentinel-1/2, HLS), the system normalizes the data, applies cloud-masking via GDAL/OGR, and extracts latent features.

```mermaid
sequenceDiagram
    participant U as Client
    participant IP as Ingestion Engine
    participant CV as Raster Processor (GDAL)
    participant RCLIP as RemoteCLIP Embedder
    participant PRITHVI as Prithvi-EO-2.0
    participant DB as SQLite / FAISS

    U->>IP: Upload Archive (ZIP/TIF/Sentinel)
    IP->>CV: Parse Telemetry & CRS
    CV->>CV: Apply Cloud Mask & Normalize Bands

    IP->>PRITHVI: Extract Geospatial Features
    PRITHVI-->>IP: Dense Feature Map

    IP->>RCLIP: Generate Visual Embeddings
    RCLIP-->>IP: 512D Vector Representation

    IP->>DB: Upsert Metadata (SQLite) & Vectors (FAISS)
    IP-->>U: Ingestion Complete
```

---

## 5. Sample Data

### Processed Sentinel-2 Imagery
*Sample scene successfully ingested, normalized, and semantically tagged:*

![Sample Sentinel-2 Data](data/imagery/sentinel2/sentinel2_scene_1.png)

---

## 6. Local Setup & Installation

### Prerequisites
* **OS:** Windows, Linux, or macOS
* **Python:** 3.11 – 3.13
* **Node.js:** 18+
* **Hardware:** 16 GB+ RAM (GPU recommended for model inference)

### Backend Setup

```bash
git clone https://github.com/your-org/geosearch-ai.git
cd geosearch-ai

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Stage Foundation Models

```bash
python scripts/stage_models.py
```

*Downloads RemoteCLIP, Prithvi-EO-2.0, SkySense++, and ChangeFormer to `data/models/`*

### Start the API Server

```bash
python run_server.py
```

* **Web UI:** `http://127.0.0.1:5173`
* **REST API Docs:** `http://127.0.0.1:8000/docs`

---

## 7. Production Deployment (Hugging Face Spaces)

GeoSearch AI can be deployed to Hugging Face Spaces using the **Docker SDK**.

1. **Hardware:** Minimum **T4 Small GPU** (A10G recommended)
2. **Persistent Storage:** Attach a Persistent Volume for `sqlite` and FAISS indexes
3. **Model Management:** Commit large models to a private HF Dataset repo via `git lfs`; configure Docker `CMD` to run `snapshot_download` on boot
4. **Environment Variables:**
   * `HF_HUB_OFFLINE=1`
   * `TRANSFORMERS_OFFLINE=1`
   * `SECRET_KEY=[secure_hmac_token]`

*Refer to `huggingface_deployment.md` for the full CI/CD guide.*
