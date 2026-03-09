from __future__ import annotations

from .affordance_classifier import AffordancePrediction, WeakAffordanceClassifier
from .affordance_features import VisualAffordanceCandidate, VisualAffordanceFeatureExtractor
from .affordance_training import AffordanceLabelDataset, AffordanceLabelExample, AffordanceLabelTarget, AffordanceTrainingSummary, AffordanceWeightTrainer
from .aligner import VLSOAligner
from .concept_dataset import VisualConceptDataset, VisualConceptExample, VisualConceptTarget
from .concept_learning import VisualConceptLabelRecommender, VisualConceptLearningSummary, VisualConceptPrototypeTrainer
from .concept_memory import VisualConceptMatch, VisualConceptMemory, VisualConceptRecord
from .data_collection import VisualCollectionPlanItem, VisualCollectionRecord, VisualCollectionRunSummary, VisualCollectionSource, VisualDataCollector, VisualDownloadEntry, VisualDownloadSummary
from .detector_adapters import DetectorOutputAdapter
from .embedding_store import VisualEmbeddingMatch, VisualEmbeddingRecord, VisualEmbeddingStore
from .geometry_reasoner import GeometryReasoningResult, VisualGeometryReasoner
from .geometry_topology import GeometryTopologyExtractor, GeometryTopologyResult
from .image_parser import ImageParseResult, RawImageObservationParser
from .image_preprocess import ImageMaskPreprocessor, ImagePreprocessResult
from .object_reasoner import VisualObjectReasoner, VisualObjectReasoningResult
from .open_images import OpenImagesAnnotationAdapter, OpenImagesPayloadSummary
from .language_parser import VLSOLanguageParser
from .operator_registry import OperatorType, VLSO_OPERATOR_TYPES
from .qa import LocalTextGenerator, VLSOAnswer, VLSOQuestionAnswerer
from .reasoner import VLSOReasoner
from .types import SharedWorldModel, VisualObservation, VLSOEntity, VLSOOperator, VLSORelation
from .vision_backbones import DEFAULT_VISION_MODEL_ROOT, VISION_BACKBONE_SPECS, VisionBackboneSpec, VisionEmbeddingExtractor, resolve_local_vision_model_path
from .visual_parser import VLSOVisualParser

__all__ = [
    "AffordanceLabelDataset",
    "AffordanceLabelExample",
    "AffordanceLabelTarget",
    "AffordancePrediction",
    "AffordanceTrainingSummary",
    "AffordanceWeightTrainer",
    "DEFAULT_VISION_MODEL_ROOT",
    "VisualConceptDataset",
    "VisualConceptExample",
    "VisualConceptMatch",
    "VisualConceptMemory",
    "VisualConceptRecord",
    "VisualConceptTarget",
    "VisualCollectionPlanItem",
    "VisualCollectionRecord",
    "VisualCollectionRunSummary",
    "VisualCollectionSource",
    "VisualDataCollector",
    "VisualDownloadEntry",
    "VisualDownloadSummary",
    "VisualConceptLabelRecommender",
    "VisualConceptLearningSummary",
    "VisualConceptPrototypeTrainer",
    "DetectorOutputAdapter",
    "GeometryReasoningResult",
    "GeometryTopologyExtractor",
    "GeometryTopologyResult",
    "ImageParseResult",
    "ImageMaskPreprocessor",
    "ImagePreprocessResult",
    "LocalTextGenerator",
    "OpenImagesAnnotationAdapter",
    "OpenImagesPayloadSummary",
    "OperatorType",
    "RawImageObservationParser",
    "SharedWorldModel",
    "VLSOAnswer",
    "VisualObjectReasoner",
    "VisualObjectReasoningResult",
    "VisualEmbeddingMatch",
    "VisualAffordanceCandidate",
    "VisualAffordanceFeatureExtractor",
    "WeakAffordanceClassifier",
    "VisualEmbeddingRecord",
    "VisualEmbeddingStore",
    "VisualGeometryReasoner",
    "VisualObservation",
    "VLSOAligner",
    "VLSOQuestionAnswerer",
    "VLSOEntity",
    "VLSOLanguageParser",
    "VLSOOperator",
    "VLSOReasoner",
    "VLSORelation",
    "VLSOVisualParser",
    "VLSO_OPERATOR_TYPES",
    "VISION_BACKBONE_SPECS",
    "VisionBackboneSpec",
    "VisionEmbeddingExtractor",
    "resolve_local_vision_model_path",
]
