from .baseline_runner import BASELINE_SPECS, BaselineRunner, BaselineSpec
from .context_chunks import TextChunk, split_context_into_chunks
from .cp_knowledge import CpAlgorithmKnowledge, CpDslOperator, CpKnowledgeBase, CpKnowledgeLoader, CpLogicalFrame
from .cp_corpus import CpCorpusBuilder, CpCorpusRecord
from .cp_dataset import CpDslExample, load_cp_dsl_examples, save_cp_dsl_examples
from .cp_episode_ingest import CpIncidentCase, CpIncidentDataset, CpIncidentIngestor
from .cp_episode_store import CpEpisodeMatch, CpEpisodeRecord, CpEpisodeStore
from .cp_parser_eval import CpLearnedParser, CpParserEvaluator, CpParserEvalSummary, CpParserPrediction
from .cp_training import CpDslDatasetBuilder, CpParserTrainConfig, CpParserTrainingScaffold, CpSftRecord, CpTrainingBundle, CpTrainingBundleBuilder, CpTrainingPlanner, load_cp_sft_records
from .cp_validation import CpSampleCase, CpSolutionValidator, CpValidationReport, CppBuildResult, CppRunResult, CppProgramRunner
from .cp_repair import CpRepairAttempt, CppRepairEngine
from .contest_programmer import CompetitiveProgrammingReasoner, ContestSolution
from .corpus_builder import CorpusBuilder
from .corpus_learning import CorpusReasoningLearner
from .corpus_store import CorpusMemoryStore
from .curated_datasets import CURATED_PRESETS, CURATED_PUBLIC_DATASETS, curated_manifest, preset_manifest
from .cpp_compiler import CppCompileResult, CppSyntaxChecker
from .dataset_adapters import PublicDatasetAdapter
from .domain_copilot import AuditItem, CopilotRequest, CopilotResult, DomainCopilot
from .domain_templates import DOMAIN_TEMPLATES
from .emergent_operators import EmergentOperatorInducer
from .feedback_rules import FeedbackRule, FeedbackRuleSet
from .hard_problem_engine import HardProblemCandidate, HardProblemEngine, HardProblemReport, PatternOutcomeTrainer, VerificationCheck
from .labeled_eval import LabeledOpsCase, LabeledOpsCaseResult, LabeledOpsEvaluator, load_labeled_ops_cases
from .logical_grammar import GrammarInductionResult, LogicalGrammarInducer, LogicalPattern, LogicalPatternMatch, LogicalPatternMatcher
from .memory_prior_eval import MemoryPriorEvaluationResult, MemoryPriorEvaluator, QueryPriorEffect
from .model_cache import DEFAULT_EMBEDDING_MODEL_ID, EmbeddingModelCache, resolve_embedding_model_id
from .olympiad_reasoner import OlympiadReasoner
from .operator_hierarchy import OperatorCompositionPattern, OperatorHierarchyLearner, OperatorHierarchyNode, OperatorHierarchyResult
from .operator_registry import RegistryNode, TypedOperatorRegistry
from .ops_kpi import OpsKpiEvaluator, OpsKpiReport
from .pipeline import StructuredMeaningPipeline
from .product_profiles import DOMAIN_PROFILES, DomainProfile, resolve_domain_profile
from .public_corpus_pipeline import ManifestEntry, ManifestRunSummary, PublicCorpusIngestor
from .rag_baseline import PlainRagBaseline, RagBaselineResult
from .remote_datasets import DownloadedArtifact, RemoteDatasetDownloader
from .response_synthesizer import ResponseSynthesizer, SynthesizedResponse
from .review_queue import ReviewQueueItem, ReviewQueueStore, review_reasons_from_kpis
from .symbolic_arithmetic import ArithmeticReasoner
from .symbolic_document import DocumentEvidenceReasoner
from .symbolic_reasoners import SymbolicReasoner
from .structures import SymbolicResult
from .transfer_eval import TransferEvaluator
from .vlso import AffordanceLabelDataset, AffordanceLabelExample, AffordanceLabelTarget, AffordancePrediction, AffordanceTrainingSummary, AffordanceWeightTrainer, DEFAULT_VISION_MODEL_ROOT, DetectorOutputAdapter, GeometryReasoningResult, GeometryTopologyExtractor, GeometryTopologyResult, ImageMaskPreprocessor, ImageParseResult, ImagePreprocessResult, LocalTextGenerator, OpenImagesAnnotationAdapter, OpenImagesPayloadSummary, OperatorType, RawImageObservationParser, SharedWorldModel, VisualAffordanceCandidate, VisualAffordanceFeatureExtractor, VisualCollectionPlanItem, VisualCollectionRecord, VisualCollectionRunSummary, VisualCollectionSource, VisualConceptDataset, VisualConceptExample, VisualConceptLabelRecommender, VisualConceptLearningSummary, VisualConceptMatch, VisualConceptMemory, VisualConceptPrototypeTrainer, VisualConceptRecord, VisualConceptTarget, VisualDataCollector, VisualDownloadEntry, VisualDownloadSummary, VisualEmbeddingMatch, VisualEmbeddingRecord, VisualEmbeddingStore, VisualGeometryReasoner, VisualObjectReasoner, VisualObjectReasoningResult, VisualObservation, VISION_BACKBONE_SPECS, VLSOAligner, VLSOAnswer, VLSOEntity, VLSOLanguageParser, VLSOOperator, VLSOQuestionAnswerer, VLSOReasoner, VLSORelation, VLSOVisualParser, VLSO_OPERATOR_TYPES, VisionBackboneSpec, VisionEmbeddingExtractor, WeakAffordanceClassifier, resolve_local_vision_model_path

__all__ = [
    "AffordanceLabelDataset",
    "AffordanceLabelExample",
    "AffordanceLabelTarget",
    "AffordancePrediction",
    "AffordanceTrainingSummary",
    "AffordanceWeightTrainer",
    "ArithmeticReasoner",
    "AuditItem",
    "BASELINE_SPECS",
    "DEFAULT_VISION_MODEL_ROOT",
    "CURATED_PRESETS",
    "CURATED_PUBLIC_DATASETS",
    "CompetitiveProgrammingReasoner",
    "ContestSolution",
    "CpCorpusBuilder",
    "CpCorpusRecord",
    "CpAlgorithmKnowledge",
    "CpKnowledgeBase",
    "CpDslExample",
    "CpDslDatasetBuilder",
    "CpIncidentCase",
    "CpIncidentDataset",
    "CpIncidentIngestor",
    "CpEpisodeMatch",
    "CpEpisodeRecord",
    "CpEpisodeStore",
    "CpParserEvalSummary",
    "CpParserEvaluator",
    "CpParserPrediction",
    "CpParserTrainConfig",
    "CpParserTrainingScaffold",
    "CpLearnedParser",
    "CpSftRecord",
    "CpTrainingBundle",
    "CpTrainingBundleBuilder",
    "CpTrainingPlanner",
    "CpRepairAttempt",
    "CppRepairEngine",
    "CpSampleCase",
    "CpValidationReport",
    "CppBuildResult",
    "CppRunResult",
    "CppProgramRunner",
    "CpDslOperator",
    "CpLogicalFrame",
    "CpKnowledgeLoader",
    "CppCompileResult",
    "CppSyntaxChecker",
    "CopilotRequest",
    "CopilotResult",
    "CorpusBuilder",
    "CorpusMemoryStore",
    "CorpusReasoningLearner",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DOMAIN_PROFILES",
    "DOMAIN_TEMPLATES",
    "DomainCopilot",
    "DomainProfile",
    "DocumentEvidenceReasoner",
    "DownloadedArtifact",
    "EmbeddingModelCache",
    "EmergentOperatorInducer",
    "FeedbackRule",
    "FeedbackRuleSet",
    "LocalTextGenerator",
    "LogicalPatternMatcher",
    "LogicalPatternMatch",
    "VerificationCheck",
    "PatternOutcomeTrainer",
    "HardProblemReport",
    "HardProblemEngine",
    "HardProblemCandidate",
    "LabeledOpsCase",
    "LabeledOpsCaseResult",
    "LabeledOpsEvaluator",
    "LogicalPattern",
    "LogicalGrammarInducer",
    "GrammarInductionResult",
    "ManifestEntry",
    "ManifestRunSummary",
    "MemoryPriorEvaluationResult",
    "MemoryPriorEvaluator",
    "OlympiadReasoner",
    "OperatorCompositionPattern",
    "OperatorHierarchyLearner",
    "OperatorHierarchyNode",
    "OperatorHierarchyResult",
    "OpsKpiEvaluator",
    "OpsKpiReport",
    "PlainRagBaseline",
    "PublicCorpusIngestor",
    "PublicDatasetAdapter",
    "QueryPriorEffect",
    "RagBaselineResult",
    "RegistryNode",
    "RemoteDatasetDownloader",
    "ResponseSynthesizer",
    "ReviewQueueItem",
    "ReviewQueueStore",
    "StructuredMeaningPipeline",
    "SymbolicReasoner",
    "SymbolicResult",
    "SynthesizedResponse",
    "TextChunk",
    "TransferEvaluator",
    "VLSO_OPERATOR_TYPES",
    "VLSOVisualParser",
    "VLSOQuestionAnswerer",
    "VLSORelation",
    "VLSOAnswer",
    "VLSOReasoner",
    "VLSOOperator",
    "VLSOLanguageParser",
    "VLSOEntity",
    "VLSOAligner",
    "VisualObservation",
    "VisionEmbeddingExtractor",
    "resolve_local_vision_model_path",
    "VisionBackboneSpec",
    "VISION_BACKBONE_SPECS",
    "VisualEmbeddingStore",
    "VisualEmbeddingRecord",
    "VisualEmbeddingMatch",
    "VisualAffordanceCandidate",
    "VisualCollectionPlanItem",
    "VisualCollectionRecord",
    "VisualCollectionRunSummary",
    "VisualDownloadEntry",
    "VisualDownloadSummary",
    "VisualCollectionSource",
    "VisualConceptDataset",
    "VisualConceptExample",
    "VisualConceptMatch",
    "VisualConceptMemory",
    "VisualConceptRecord",
    "VisualConceptTarget",
    "VisualDataCollector",
    "VisualConceptLabelRecommender",
    "VisualConceptLearningSummary",
    "VisualConceptPrototypeTrainer",
    "VisualAffordanceFeatureExtractor",
    "WeakAffordanceClassifier",
    "GeometryTopologyResult",
    "ImageMaskPreprocessor",
    "ImageParseResult",
    "ImagePreprocessResult",
    "GeometryTopologyExtractor",
    "VisualGeometryReasoner",
    "VisualObjectReasoner",
    "VisualObjectReasoningResult",
    "GeometryReasoningResult",
    "DetectorOutputAdapter",
    "SharedWorldModel",
    "OpenImagesAnnotationAdapter",
    "OpenImagesPayloadSummary",
    "OperatorType",
    "RawImageObservationParser",
    "TypedOperatorRegistry",
    "BaselineRunner",
    "BaselineSpec",
    "curated_manifest",
    "load_cp_dsl_examples",
    "load_cp_sft_records",
    "load_labeled_ops_cases",
    "save_cp_dsl_examples",
    "preset_manifest",
    "resolve_domain_profile",
    "resolve_embedding_model_id",
    "review_reasons_from_kpis",
    "split_context_into_chunks",
]








