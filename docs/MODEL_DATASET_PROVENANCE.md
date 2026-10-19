# Model & Dataset Provenance Declaration

**Project:** GeoSearch AI  
**Deployment Context:** Air-gapped / Local Sovereign On-Premises Execution  
**Organization:** Ministry of Defence (MoD) / Indian Army (DGIS)  
**Standard Compliance:** Fully offline operation, zero cloud calls during inference.

---

## 1. Foundation Models Provenance

All foundation models employed in GeoSearch AI are approved open-weight architectures, pre-staged locally for air-gapped evaluation.

| Component | Model & Architecture | Parameter Count | Source Repository | License | Weight Checksum / Format | Representational Axis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Semantic Retrieval** | **RemoteCLIP** (ViT-L/14) | ~304M | `chendelong/RemoteCLIP` (IEEE TGRS) | MIT | OpenCLIP PyTorch (`.pt`) | Natural language free-text & image-to-image similarity |
| **Temporal Sequence** | **Prithvi-EO-2.0** (3D ViT) | ~100M - 300M | `ibm-nasa-geospatial/Prithvi-100M` | Apache-2.0 | HuggingFace Safetensors | Multi-temporal spectral evolution & change onset |
| **Multi-Sensor Fusion** | **SkySense++** (Cross-Attention) | ~200M | `AI4EO/SkySense` (Nature Mach. Intell.) | Apache-2.0 | Dual-Branch Safetensors | Co-registered Sentinel-1 SAR & Sentinel-2 Optical alignment |
| **Change Segmentation** | **ChangeFormer** | ~41M | `wgcban/ChangeFormer` | MIT | Siamese Transformer (`.pt`) | Multi-scale pixel-level change probability heatmaps |
| **Query Understanding** | **Qwen3-0.6B** | 0.6B | `Qwen/Qwen3-0.6B` | Apache-2.0 | HuggingFace GGUF / Safetensors | Natural language filter parsing & evidence grounded Q&A |
| **Visual Captioning** | **BLIP Image Captioning** | ~224M | `Salesforce/blip-image-captioning-base` | BSD-3-Clause | PyTorch `.bin` | Automated ingestion description generation |

---

## 2. Pretraining & Benchmark Datasets

Foundation models were pretrained exclusively on open scientific Earth-observation corpuses:

| Dataset | Modality | Provider / Source | License | Associated Model |
| :--- | :--- | :--- | :--- | :--- |
| **RSICD / RSITMD** | Optical Multispectral + Captions | IEEE Geoscience & Remote Sensing Society | CC BY 4.0 | RemoteCLIP |
| **Harmonized Landsat-Sentinel (HLS)** | 30m / 10m Multi-temporal time-series | NASA LP DAAC & USGS | Public Domain / US Gov Open Data | Prithvi-EO-2.0 |
| **Copernicus Sentinel-1 SAR** | C-band Synthetic Aperture Radar (GRD) | European Space Agency (ESA) | Copernicus Open Access License | SkySense++ |
| **Copernicus Sentinel-2 MSI** | 13-band Optical / SWIR Surface Reflectance | European Space Agency (ESA) | Copernicus Open Access License | SkySense++, ChangeFormer |
| **ISRO Bhuvan Open Products** | Optical Imagery Archives | ISRO / National Remote Sensing Centre (NRSC) | Open Government Data (OGD) India | Evaluation Archive |

---

## 3. Physical Preprocessing & Ingestion Provenance

1. **Spectral Indices Computed at Chip Ingestion:**
   - **NDVI:** $(NIR - Red) / (NIR + Red)$
   - **NDWI:** $(Green - NIR) / (Green + NIR)$
   - **NDBI:** $(SWIR - NIR) / (SWIR + NIR)$
   - **EVI:** $2.5 \times (NIR - Red) / (NIR + 6Red - 7.5Blue + 1)$
2. **Atmospheric & Illumination Handling:** Dark Object Subtraction (DOS) and Cumulative Distribution Function (CDF) histogram matching across temporal acquisitions.
3. **Geometric Fidelity:** Affine geotransform matrices and EPSG Coordinate Reference Systems (CRS) preserved for every $256\times256$ tile chip.
4. **Sub-Pixel Co-Registration:** Normalized Cross-Correlation (NCC) / Enhanced Correlation Coefficient (ECC) alignment computed per bi-temporal pair.
5. **SAR Preprocessing:** Radiometric calibration to Sigma-0 backscatter ($\text{dB}$) followed by adaptive $5\times5$ Lee speckle filtering.

---

## 4. Evaluation & Air-Gap Compliance Statement

- **Network Isolation:** GeoSearch AI operates with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **Zero API Telemetry:** No external network sockets or third-party cloud APIs are queried during ingestion, embedding, FAISS search, or change inference.
- **Audit Logging:** Every analyst action (Confirm, Reject, Re-label) and automated pipeline transformation step is recorded in SQLite with cryptographic SHA-256 hashes of tile artifacts and model descriptors.
