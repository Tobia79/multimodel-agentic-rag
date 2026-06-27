"""Document ingestion pipeline: parse → clean → chunk → enrich → index."""

from ingestion.pipeline import IngestionPipeline, PipelineResult, run_pipeline

__all__ = ["IngestionPipeline", "PipelineResult", "run_pipeline"]
