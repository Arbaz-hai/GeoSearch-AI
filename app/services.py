from __future__ import annotations

import logging
from dataclasses import dataclass

from app.conversation.query_parser import QueryParser
from app.embeddings.prithvi_encoder import PrithviEncoder
from app.embeddings.skysense_encoder import SkySenseEncoder
from app.embeddings.text_image_encoder import TextImageEncoder
from app.embeddings.visual_encoder import VisualEncoder
from app.index.faiss_store import FaissStore
from app.index.metadata_store import MetadataStore
from app.index.qdrant_store import QdrantStore
from app.pipelines.change_pipeline import ChangePipeline
from app.pipelines.cluster_pipeline import ClusterPipeline
from app.pipelines.export_pipeline import ExportPipeline
from app.pipelines.ingest_pipeline import IngestPipeline
from app.pipelines.search_pipeline import SearchPipeline
from app.settings import Settings
from app.training_data.export_dataset import FeedbackDataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ApplicationServices:
    settings: Settings
    metadata: MetadataStore
    vectors: QdrantStore
    faiss: FaissStore
    text_encoder: TextImageEncoder
    visual_encoder: VisualEncoder
    prithvi: PrithviEncoder
    skysense: SkySenseEncoder
    query_parser: QueryParser
    ingest: IngestPipeline
    search: SearchPipeline
    change: ChangePipeline
    cluster: ClusterPipeline
    export: ExportPipeline
    feedback: FeedbackDataset


def create_services(settings: Settings) -> ApplicationServices:
    logger.info("GeoSearch AI — initialising services")

    metadata = MetadataStore(settings.database_path)
    vectors = QdrantStore(
        settings.qdrant_path,
        settings.semantic_collection,
        settings.visual_collection,
        settings.embedding_dimension,
    )
    faiss = FaissStore(
        settings.index_dir / "faiss",
        dimension=settings.embedding_dimension,
    )

    clip_path = settings.model_path("clip")
    remoteclip_path = settings.model_path("remoteclip")
    encoder_path = clip_path if (clip_path.exists() and any(clip_path.iterdir()) if clip_path.exists() else False) else remoteclip_path
    text_encoder = TextImageEncoder(
        encoder_path, settings.embedding_dimension, settings.offline_mode
    )
    logger.info("Text/image encoder backend: %s", text_encoder.backend)

    visual_encoder = VisualEncoder(
        settings.model_path("prithvi"), settings.embedding_dimension, settings.offline_mode
    )
    logger.info("Visual encoder backend: %s", visual_encoder.backend)

    prithvi = PrithviEncoder(
        settings.model_path("prithvi"), settings.embedding_dimension, settings.offline_mode
    )
    logger.info("Prithvi temporal encoder backend: %s", prithvi.backend)

    skysense = SkySenseEncoder(
        settings.model_path("skysense"), settings.embedding_dimension, settings.offline_mode
    )
    logger.info("SkySense++ SAR-Optical fusion encoder backend: %s", skysense.backend)

    query_parser = QueryParser(settings.model_path("qwen3"), settings.offline_mode)
    logger.info("Query parser backend: %s", query_parser.backend)

    search = SearchPipeline(settings, metadata, vectors, text_encoder, visual_encoder, query_parser)

    services = ApplicationServices(
        settings=settings,
        metadata=metadata,
        vectors=vectors,
        faiss=faiss,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        prithvi=prithvi,
        skysense=skysense,
        query_parser=query_parser,
        ingest=IngestPipeline(settings, metadata, vectors, text_encoder, visual_encoder),
        search=search,
        change=ChangePipeline(settings, metadata, vectors),
        cluster=ClusterPipeline(settings, metadata, vectors, search),
        export=ExportPipeline(metadata, settings.root_dir / "output"),
        feedback=FeedbackDataset(settings.data_dir / "training_data"),
    )

    _log_model_status(settings)
    return services


def _log_model_status(settings: Settings) -> None:
    """Log which models are staged and which are missing."""
    model_names = ["clip", "blip", "qwen3", "remoteclip", "prithvi", "skysense", "changeformer"]
    staged = []
    missing = []
    for name in model_names:
        path = settings.model_path(name)
        if path.exists() and any(path.iterdir() if path.is_dir() else [True]):
            staged.append(name)
        else:
            missing.append(name)

    if staged:
        logger.info("Staged models: %s", ", ".join(staged))
    if missing:
        logger.warning(
            "Missing model weights (running in fallback mode): %s  "
            "Run: python scripts/stage_models.py  to download them.",
            ", ".join(missing),
        )
