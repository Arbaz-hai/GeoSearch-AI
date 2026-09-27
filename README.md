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

GeoSearch AI is built around four intelligence modules that operate on Earth Observation inputs (Sentinel-2/HLS optical, Sentinel-1 SAR GRD, and multi-date temporal archives) and deliver structured analyst outputs.

---

### 4.1 Module 1 — Semantic Intelligence *(Optical + Text)*

Natural language queries and optical tiles are independently encoded by **RemoteCLIP ViT-L/14** into a shared 512-D embedding space. A **FAISS Semantic Index** performs hybrid retrieval combining Cosine Similarity, spectral relevance, and metadata filters to return ranked image candidates.

```mermaid
flowchart LR
    NLQ["Natural Language Query\n(e.g. 'new structures near river')"]
    OPT["Optical Tiles\n(Sentinel-2 / HLS / GeoTIFF-COG)"]
    PRE["Preprocessing\nRGB / False Color"]
    TXT_ENC["RemoteCLIP\nText Encoder"]
    VIS_ENC["RemoteCLIP\nViT-L/14\nVisual Encoder"]
    TVEC["512-D Text Embedding"]
    VVEC["512-D Visual Embedding"]
    FAISS[("FAISS\nSemantic Index")]
    HYBRID["Hybrid Retrieval\nCosine Similarity +\nSpectral Relevance +\nMetadata"]
    OUT["Ranked Candidates"]

    NLQ --> TXT_ENC --> TVEC --> FAISS
    OPT --> PRE --> VIS_ENC --> VVEC --> FAISS
    FAISS --> HYBRID --> OUT
```

---

### 4.2 Module 2 — Temporal Intelligence *(Multi-Temporal Analysis)*

Multi-temporal image sequences are co-registered via **ECC + Histogram Matching**, then encoded by **Prithvi-EO-2.0** (3D ViT Spatiotemporal Encoder) into temporal embedding sequences. Two parallel heads then operate:

- **ChangeFormer** (Siamese Transformer) → pixel-level change mask
- **Binary Search O(log N)** → earliest change detection with temporal onset

```mermaid
flowchart LR
    SEQ["Multi-Temporal\nImage Sequence"]
    HIST["Histogram\nMatching"]
    ECC["ECC\nCo-Registration"]
    PRITHVI["Prithvi-EO-2.0\n3D ViT /\nSpatiotemporal Encoder"]
    TEMB["Temporal Embedding\nSequence"]
    CF["ChangeFormer\nSiamese Transformer"]
    BS["Binary Search\nO(log N)"]
    MASK["Pixel-Level\nChange Mask"]
    EC["Earliest\nChange"]

    SEQ --> HIST --> ECC --> PRITHVI --> TEMB
    TEMB --> CF --> MASK
    TEMB --> BS --> EC
```

---

### 4.3 Module 3 — Multi-Sensor Fusion *(SAR + Optical)*

Sentinel-1 SAR data undergoes radiometric calibration and Lee Speckle filtering. It is then fused with co-registered Sentinel-2 optical data via **SkySense++** (SAR-Optical Cross-Attention), producing a fused 512-D embedding stored in a dedicated **SAR-Optical FAISS Index**.

```mermaid
flowchart LR
    SAR["Sentinel-1 SAR\n(GRD)"]
    OPT2["Sentinel-2\nOptical"]
    RCAL["Radiometric\nCalibration"]
    LEE["Lee Speckle\nFilter"]
    COREG["Co-Registration"]
    SKYSENSE["SkySense++\nSAR-Optical\nCross-Attention"]
    FUSED["Fused 512-D\nEmbedding"]
    SARFAISS[("SAR-Optical\nFAISS Index")]

    SAR --> RCAL --> LEE --> SKYSENSE
    OPT2 --> COREG --> SKYSENSE
    SKYSENSE --> FUSED --> SARFAISS
```

---

### 4.4 Module 4 — False-Alarm / Evidence Layer

Before a change is confirmed, a **Multi-Evidence Gate** validates it against multiple physical signals to suppress false alarms:

| Evidence Signal | Purpose |
|---|---|
| Change Mask | Primary ChangeFormer output |
| NDVI / NDWI / NDBI / EVI Deltas | Vegetation, water, built-up, vegetation index changes |
| SCL Cloud Mask | Suppress cloud-induced artifacts |
| Shadow / DSM | Terrain and shadow interference filtering |
| Radiometric Normalization | Cross-date illumination correction |

Only detections passing all gates are emitted as **Confidence-Gated Changes**.

```mermaid
flowchart TD
    CMASK["Change Mask"]
    NDVI["NDVI / NDWI /\nNDBI / EVI Deltas"]
    SCL["SCL Cloud Mask"]
    SHADOW["Shadow / DSM"]
    RADNORM["Radiometric\nNormalization"]
    GATE["Multi-Evidence Gate\nAI Evidence + Physical Evidence"]
    OUT2["Confidence-Gated Change ✅"]

    CMASK --> GATE
    NDVI --> GATE
    SCL --> GATE
    SHADOW --> GATE
    RADNORM --> GATE
    GATE --> OUT2
```

---

### 4.5 Intelligence Core & Active Learning

All three FAISS indexes (Visual, Temporal, SAR-Optical) are managed by the **GeoSearch AI Intelligence Core**. Analyst feedback (Confirm / Reject / Re-label) flows into an **Active Learning** loop that adjusts FAISS confidence modifiers — with no foundation-model retraining required.

```mermaid
flowchart LR
    CORE["GeoSearch AI\nIntelligence Core"]
    SEM["SEMANTIC\nRemoteCLIP"]
    TEMP["TEMPORAL\nPrithvi-EO-2.0"]
    SENS["SENSOR FUSION\nSkySense++"]
    VI[("Visual\nIndex")]
    TI[("Temporal\nIndex")]
    SI[("SAR-Optical\nIndex")]
    ANALYST["Analyst Review\nConfirm / Reject / Re-label"]
    AL["Active Learning\nFAISS confidence modifiers"]
    GEO["GeoJSON Export\nSource Scene + Model Hash +\nAudit Trail + Timestamp"]

    CORE --> SEM & TEMP & SENS
    SEM --> VI
    TEMP --> TI
    SENS --> SI
    VI & TI & SI --> ANALYST
    ANALYST --> AL --> CORE
    ANALYST --> GEO
```

---

## 5. Local Setup & Installation

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

## 6. Production Deployment (Hugging Face Spaces)

GeoSearch AI can be deployed to Hugging Face Spaces using the **Docker SDK**.

1. **Hardware:** Minimum **T4 Small GPU** (A10G recommended)
2. **Persistent Storage:** Attach a Persistent Volume for `sqlite` and FAISS indexes
3. **Model Management:** Commit large models to a private HF Dataset repo via `git lfs`; configure Docker `CMD` to run `snapshot_download` on boot
4. **Environment Variables:**
   * `HF_HUB_OFFLINE=1`
   * `TRANSFORMERS_OFFLINE=1`
   * `SECRET_KEY=[secure_hmac_token]`

*Refer to `huggingface_deployment.md` for the full CI/CD guide.*
