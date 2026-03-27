from __future__ import annotations

from .affordance_classifier import AffordancePrediction, WeakAffordanceClassifier
from .affordance_features import VisualAffordanceCandidate, VisualAffordanceFeatureExtractor
from .affordance_training import AffordanceLabelDataset, AffordanceLabelExample, AffordanceLabelTarget, AffordanceTrainingSummary, AffordanceWeightTrainer
from .aligner import VLSOAligner
from ..operator_algebra import FunctorHypothesis, OperatorAlgebraLearner, OperatorAlgebraSummary, OperatorDecomposition
from .concept_dataset import VisualConceptDataset, VisualConceptExample, VisualConceptTarget
from .concept_learning import VisualConceptLabelRecommender, VisualConceptLearningSummary, VisualConceptPrototypeTrainer
from .concept_memory import VisualConceptMatch, VisualConceptMemory, VisualConceptRecord
from .data_collection import VisualCollectionPlanItem, VisualCollectionRecord, VisualCollectionRunSummary, VisualCollectionSource, VisualDataCollector, VisualDownloadEntry, VisualDownloadSummary, VisualFamilyBatchSummary, build_object_family_manifest, build_geometry_seed_manifest
from .eval import VlsoEvalCase, VlsoEvalResult, VlsoEvalSummary, VlsoGroundedEvaluator
from .geometry_dataset import SyntheticGeometryScene, SyntheticGeometrySceneBuilder
from .geometry_pipeline import GeometryPipelineSummary, VisualGeometryBootstrapPipeline
from .detector_adapters import DetectorOutputAdapter
from .embedding_store import VisualEmbeddingMatch, VisualEmbeddingRecord, VisualEmbeddingStore
from .frontier_setup import FrontierBundleItem, FrontierInstallSummary, FrontierSetupSummary, FrontierVisionInstaller
from .frontier_vlm import FrontierVisualSummary, FrontierVisionAdapter, FrontierVisionSpec, VisualSceneAdjudication, VisualSceneAdjudicator
from .geometry_backbones import GeometryPrimitiveBackbone, GeometryPrimitiveResult
from .geometry_reasoner import GeometryReasoningResult, VisualGeometryReasoner
from .hybrid_memory import HybridMemoryMatch, VisualHybridMemory, VisualHybridMemoryResult
from .geometry_topology import GeometryTopologyExtractor, GeometryTopologyResult
from .image_parser import ImageParseResult, RawImageObservationParser
from .image_preprocess import ImageMaskPreprocessor, ImagePreprocessResult
from .object_reasoner import VisualObjectReasoner, VisualObjectReasoningResult
from .open_images import OpenImagesAnnotationAdapter, OpenImagesPayloadSummary
from .language_parser import VLSOLanguageParser
from .operator_registry import OperatorType, VLSO_OPERATOR_TYPES
from .operator_learning import VisualOperatorLearningSummary, VisualOperatorMatch, VisualOperatorMemory, VisualOperatorPrototypeTrainer, VisualOperatorRecord
from .predictive_priors import JepaStructuralPredictor, PredictivePriorResult
from .qa import LocalTextGenerator, VLSOAnswer, VLSOQuestionAnswerer
from .semantic_scene import SemanticRegionHypothesis, SemanticSceneAnalyzer, SemanticSceneHypothesis, SemanticSceneSummary
from .self_training import PseudoLabelAcceptanceConfig, VisualConceptSelfTrainer, VisualPseudoCluster, VisualSelfTrainingSummary
from .cluster_review import VisualClusterReviewDecision, VisualClusterReviewStore
from .review_retrain import VisualApprovedReviewRetrainer, VisualReviewRetrainSummary
from .review_eval import VlsoReviewImpactEvaluator, VlsoStoreComparisonSummary
from .structural_operators import StructuralOperatorBinding, VisualStructuralOperatorInducer, VisualStructuralReasoningResult
from .reasoner import VLSOReasoner
from .types import SharedWorldModel, VisualObservation, VLSOEntity, VLSOOperator, VLSORelation
from .vision_backbones import DEFAULT_VISION_MODEL_ROOT, VISION_BACKBONE_SPECS, VisionBackboneSpec, VisionEmbeddingExtractor, resolve_local_vision_model_path
from .visual_parser import VLSOVisualParser

__all__ = [
    "AffordanceLabelDataset",
    "FunctorHypothesis",
    "OperatorAlgebraLearner",
    "OperatorAlgebraSummary",
    "OperatorDecomposition",
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
    "VisualFamilyBatchSummary",
    "build_object_family_manifest",
    "build_geometry_seed_manifest",
    "VlsoEvalCase",
    "VlsoEvalResult",
    "VlsoEvalSummary",
    "VlsoGroundedEvaluator",
    "SyntheticGeometryScene",
    "SyntheticGeometrySceneBuilder",
    "GeometryPipelineSummary",
    "VisualGeometryBootstrapPipeline",
    "VisualConceptLabelRecommender",
    "VisualConceptLearningSummary",
    "VisualConceptPrototypeTrainer",
    "VisualConceptSelfTrainer",
    "VisualClusterReviewDecision",
    "VisualClusterReviewStore",
    "VisualApprovedReviewRetrainer",
    "VisualReviewRetrainSummary",
    "StructuralOperatorBinding",
    "VisualStructuralOperatorInducer",
    "VisualStructuralReasoningResult",
    "VlsoReviewImpactEvaluator",
    "VlsoStoreComparisonSummary",
    "VisualPseudoCluster",
    "VisualSelfTrainingSummary",
    "DetectorOutputAdapter",
    "FrontierBundleItem",
    "FrontierInstallSummary",
    "FrontierSetupSummary",
    "FrontierVisionInstaller",
    "FrontierVisualSummary",
    "FrontierVisionAdapter",
    "FrontierVisionSpec",
    "VisualSceneAdjudication",
    "VisualSceneAdjudicator",
    "GeometryPrimitiveBackbone",
    "GeometryPrimitiveResult",
    "GeometryReasoningResult",
    "GeometryTopologyExtractor",
    "GeometryTopologyResult",
    "ImageParseResult",
    "ImageMaskPreprocessor",
    "ImagePreprocessResult",
    "LocalTextGenerator",
    "SemanticRegionHypothesis",
    "SemanticSceneAnalyzer",
    "SemanticSceneHypothesis",
    "SemanticSceneSummary",
    "OpenImagesAnnotationAdapter",
    "OpenImagesPayloadSummary",
    "PseudoLabelAcceptanceConfig",
    "OperatorType",
    "RawImageObservationParser",
    "SharedWorldModel",
    "VLSOAnswer",
    "VisualObjectReasoner",
    "VisualObjectReasoningResult",
    "VisualOperatorLearningSummary",
    "VisualOperatorMatch",
    "VisualOperatorMemory",
    "VisualOperatorPrototypeTrainer",
    "VisualOperatorRecord",
    "JepaStructuralPredictor",
    "PredictivePriorResult",
    "VisualEmbeddingMatch",
    "VisualAffordanceCandidate",
    "VisualAffordanceFeatureExtractor",
    "WeakAffordanceClassifier",
    "VisualEmbeddingRecord",
    "VisualEmbeddingStore",
    "VisualGeometryReasoner",
    "HybridMemoryMatch",
    "VisualHybridMemory",
    "VisualHybridMemoryResult",
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
