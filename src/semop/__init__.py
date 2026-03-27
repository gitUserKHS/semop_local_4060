from .basis_operators import BASIS_OPERATOR_AXES, basis_operator_axis, canonicalize_basis_signature, infer_basis_operators, normalize_operator_symbol
from .baseline_runner import BASELINE_SPECS, BaselineRunner, BaselineSpec
from .context_chunks import TextChunk, split_context_into_chunks
from .context_understanding import OperatorContextAnalyzer
from .continuous_learning import ContinuousLearningBundleBuilder, ContinuousLearningBundleSummary, build_continuous_learning_bundle
from .environment_brain import EnvironmentBrainRunner, EnvironmentBrainSummary, EnvironmentConceptStat, EnvironmentRoutineStat, EnvironmentProbe
from .adaptive_environment_learning import AdaptiveActionRehearsal, AdaptiveActionStep, AdaptiveEnvironmentAxis, AdaptiveEnvironmentLearningRunner, AdaptiveEnvironmentLearningSummary
from .recursive_self_evolution import EvolvingImprovementProgram, RecursiveSelfEvolutionRunner, RecursiveSelfEvolutionSummary, SelfEvolutionGenerationSummary
from .graph_supervision import GraphSupervisionExample, GraphSupervisionExporter, GraphSupervisionExportSummary, export_graph_supervision_from_graphs
from .analogy_policy import AnalogyPolicyModel, AnalogyPolicyScorer, AnalogyPolicyTrainer, AnalogyPolicyTrainingSummary
from .unified_parser import LearnedUnifiedParser, UnifiedParserModel, UnifiedParserTrainer, UnifiedParserTrainingSummary
from .retained_operator_algebra import RetainedOperatorAlgebra, RetainedOperatorModel, RetainedOperatorRecord, RetainedOperatorTrainer, RetainedOperatorTrainingSummary
from .operator_repair import OperatorRepairAttempt, OperatorRepairEngine
from .operator_repair_policy import OperatorRepairPolicyModel, OperatorRepairPolicyScorer, OperatorRepairPolicyTrainer, OperatorRepairPolicyTrainingSummary, train_repair_policy_from_graphs
from .repair_utility import RepairUtilityModel, RepairUtilityScorer, RepairUtilityTrainer, RepairUtilityTrainingSummary, train_repair_utility_from_graphs
from .retained_repair_programs import RetainedRepairProgramLibrary, RetainedRepairProgramModel, RetainedRepairProgramRecord, RetainedRepairProgramTrainer, RetainedRepairProgramTrainingSummary, train_retained_repair_programs_from_graphs
from .unified_benchmark import AnalogyEvalCase, BenchmarkGateDecision, BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer, BenchmarkGatedTrainingSummary, BenchmarkSliceSummary, CompilerRepairEvalCase, GroundedExplanationEvalCase, PersistentBenchmarkCorpusSummary, PromotedReviewBenchmarkCases, UnifiedBenchmarkHarness, UnifiedBenchmarkSummary, UnifiedSemOpArtifacts, UnifiedSemOpTrainer, UnifiedSemOpTrainingSummary
from .generalization_proof import GeneralizationGoalAxis, GeneralizationGoalTracker, GeneralizationProofEvidence, GeneralizationProofHarness, GeneralizationProofRound, GeneralizationProofSummary
from .grounding_self_evolution import GroundingReflection, GroundingSelfEvolutionRound, GroundingSelfEvolutionRunner, GroundingSelfEvolutionSummary
from .cp_knowledge import CpAlgorithmKnowledge, CpDslOperator, CpKnowledgeBase, CpKnowledgeLoader, CpLogicalFrame
from .cp_corpus import CpCorpusBuilder, CpCorpusRecord
from .cp_dataset import CpDslExample, load_cp_dsl_examples, save_cp_dsl_examples
from .cp_episode_ingest import CpIncidentCase, CpIncidentDataset, CpIncidentIngestor
from .cp_episode_store import CpEpisodeMatch, CpEpisodeRecord, CpEpisodeStore
from .cp_parser_eval import CpLearnedParser, CpParserEvaluator, CpParserEvalSummary, CpParserPrediction, CpParserComparisonSummary
from .cp_experiment import CpLoraExperimentConfig, CpLoraExperimentRunner, CpLoraExperimentSummary
from .cp_geometry_eval import CpGeometryEvalBuildSummary, CpGeometryEvalBuilder
from .cp_geometry_templates import CpGeometryTemplateGenerator, CpGeometryTemplateSplitSummary, CpGeometryTemplateSummary
from .cp_public_datasets import CpLabeledDatasetDownloader, CpLabeledDatasetEntry, CpLabeledDatasetSummary
from .cp_training import CpDslDatasetBuilder, CpParserTrainConfig, CpParserTrainingScaffold, CpSftRecord, CpTrainingBundle, CpTrainingBundleBuilder, CpTrainingPlanner, load_cp_sft_records
from .cp_validation import CpSampleCase, CpSolutionValidator, CpValidationReport, CppBuildResult, CppRunResult, CppProgramRunner
from .cp_repair import CpRepairAttempt, CppRepairEngine
from .contest_programmer import CompetitiveProgrammingReasoner, ContestProblemStructure, ContestSolution
from .capability_audit import CapabilityAuditRunner, CapabilityAuditSummary, CapabilityAxisStatus
from .capability_coach import CapabilityImprovementRunner, CapabilityImprovementSummary, CapabilityCurriculumRound
from .rtx4060_coach import RTX4060CollectionLane, RTX4060OptimizationSummary, RTX4060ReasoningCoach
from .concept_fusion import ConceptFusionEngine, ConceptFusionHypothesis, ConceptFusionSummary
from .ultimate_agi_readiness import UltimateAGIAxis, UltimateAGIReadinessRunner, UltimateAGIReadinessSummary
from .common_eval import SemOpCommonEvaluator, SemOpEvalSnapshot
from .corpus_builder import CorpusBuilder
from .corpus_learning import CorpusReasoningLearner
from .corpus_store import CorpusMemoryStore
from .curated_datasets import CURATED_PRESETS, CURATED_PUBLIC_DATASETS, curated_manifest, preset_manifest
from .cpp_compiler import CppCompileResult, CppSyntaxChecker
from .dataset_adapters import PublicDatasetAdapter
from .domain_copilot import AuditItem, CopilotRequest, CopilotResult, DomainCopilot
from .distillation import DistillationSftRecord, TeacherTraceExporter, TeacherTraceRecord
from .domain_templates import DOMAIN_TEMPLATES
from .emergent_operators import EmergentOperatorInducer
from .feedback_rules import FeedbackRule, FeedbackRuleSet
from .hard_problem_engine import HardProblemCandidate, HardProblemEngine, HardProblemReport, PatternOutcomeTrainer, VerificationCheck
from .hardware_profiles import LocalDependencyStatus, LocalHardwareProfile, detect_local_hardware, detect_local_ml_stack, recommended_generation_tokens, should_force_4bit
from .intelligence_map import IntelligenceAxis, IntelligenceSubsystem, OperatorIntelligenceMap, build_operator_intelligence_map
from .labeled_eval import LabeledOpsCase, LabeledOpsCaseResult, LabeledOpsEvaluator, load_labeled_ops_cases
from .leworldmodel_adapter import LeWorldModelAdapter, LeWorldModelAlignment, LeWorldModelArtifact, LeWorldModelSequenceRecord
from .leworldmodel_planner import LeWorldModelPlan, LeWorldModelPlannedAction, LeWorldModelTrajectoryPlanner
from .logical_grammar import GrammarInductionResult, LogicalGrammarInducer, LogicalPattern, LogicalPatternMatch, LogicalPatternMatcher
from .memory_prior_eval import MemoryPriorEvaluationResult, MemoryPriorEvaluator, QueryPriorEffect
from .multimodal_alignment_memory import MultimodalAlignmentMemory, MultimodalAlignmentModel, MultimodalAlignmentRecord, MultimodalAlignmentTrainer, MultimodalAlignmentTrainingSummary, train_multimodal_alignment_from_graphs
from .operating_policies import DEFAULT_REVIEW_SEVERITY_WEIGHTS, DomainOperatingPolicy, OPERATING_POLICIES, ReviewPromotionDecision, collect_review_promotion_decisions, evaluate_review_promotion, infer_operating_domain, promoted_review_details, resolve_operating_policy, resolve_review_severity_weight, resolve_slice_balance_limit, resolve_slice_thresholds
from .model_cache import DEFAULT_EMBEDDING_MODEL_ID, EmbeddingModelCache, resolve_embedding_model_id
from .olympiad_reasoner import OlympiadReasoner
from .operator_hierarchy import OperatorCompositionPattern, OperatorHierarchyLearner, OperatorHierarchyNode, OperatorHierarchyResult
from .operator_algebra import FunctorHypothesis, OperatorAlgebraLearner, OperatorAlgebraSummary, OperatorDecomposition
from .operator_algebra_eval import OperatorAlgebraEvalCase, OperatorAlgebraEvaluator, OperatorAlgebraEvalSummary
from .operator_curriculum import OperatorCurriculumBuilder, OperatorCurriculumPhase, OperatorLearningBundleSummary
from .operator_evolution import EvolvedOperatorProposal, OperatorEvolutionRunResult, OperatorEvolutionSummary, OperatorSelfEvolutionEngine, OperatorSelfEvolutionLoop, OperatorTransferEvalCase, OperatorTransferEvalSummary, OperatorTransferEvaluator
from .operator_proposal import ModelProposedOperator, OperatorProposalEngine, OperatorProposalPattern, OperatorProposalSummarizer
from .operator_proposal_eval import HybridOperatorProposalPolicy, HybridOperatorProposalSummary, OperatorProposalComparisonSummary, OperatorProposalComparator, VisualSignalImpactEvaluator, VisualSignalImpactSummary
from .progress_report import OperatorIntelligenceProgress, OperatorIntelligenceProgressEstimator, ProgressAxis
from .operator_registry import RegistryNode, TypedOperatorRegistry
from .operator_runtime import OperatorCompiler, OperatorExecutor, compile_and_execute
from .operator_training import OperatorTrainConfig, OperatorTrainingScaffold
from .premise_eval import HiddenPremiseEvalCase, HiddenPremiseEvaluator, HiddenPremiseEvalSummary
from .premise_explorer import HiddenPremiseExplorer, HiddenPremiseResult
from .ops_kpi import OpsKpiEvaluator, OpsKpiReport
from .pipeline import StructuredMeaningPipeline
from .product_profiles import DOMAIN_PROFILES, DomainProfile, resolve_domain_profile
from .public_corpus_pipeline import ManifestEntry, ManifestRunSummary, PublicCorpusIngestor
from .rag_baseline import PlainRagBaseline, RagBaselineResult
from .remote_datasets import DownloadedArtifact, RemoteDatasetDownloader
from .response_synthesizer import ResponseSynthesizer, SynthesizedResponse
from .runtime_ops import BeginnerOneClickSummary, RuntimeDoctorReport, RuntimeLaunchEntry, RuntimeLaunchSummary, RuntimeSurfaceSpec, build_runtime_doctor_report, launch_beginner_one_click, launch_runtime_stack
from .script_compatibility import ScriptCompatibilityBreakdown, ScriptCompatibilityModel, ScriptCompatibilityScorer, ScriptCompatibilityTrainer, ScriptCompatibilityTrainingSummary
from .review_queue import ReviewQueueItem, ReviewQueueStore, infer_review_severity, normalize_review_severity, review_reasons_from_graph, review_reasons_from_graph_and_kpis, review_reasons_from_kpis, severity_weight
from .prompt_understanding import PromptUnderstandingAnalyzer, PromptUnderstandingSummary
from .multimodal_scene_understanding import FrameSituationSummary, TemporalSceneReasoner, TemporalSituationSummary
from .symbolic_arithmetic import ArithmeticReasoner
from .symbolic_document import DocumentEvidenceReasoner
from .symbolic_reasoners import SymbolicReasoner
from .structures import ClaimGrounding, ContextFrame, FunctorHypothesis as GraphFunctorHypothesis, GoalPreservationCheck, OperatorDecomposition as GraphOperatorDecomposition, OperatorExecutionReport, OperatorInstruction, PremiseCandidate, PremiseValidation, SymbolicResult
from .transfer_eval import TransferEvaluator
from .turboquant_review import QuantizedReviewAssignment, TurboQuantReviewPlanner, TurboQuantReviewSummary
from .understanding_eval import SemOpUnderstandingEvaluator, UnderstandingEvalSummary
from .world_model_math import MathStrategyPrior, MathWorldCandidate, MathWorldCheck, WorldModelMathReasoner, WorldModelMathReport
from .world_model_math_service import ProductionMathDecision, ProductionMathMetrics, ProductionMathReadiness, ProductionMathSelfTestCase, ProductionMathSelfTestResult, ProductionMathSelfTestSummary, ProductionMathServiceConfig, ProductionMathServiceResponse, WorldModelMathProductionService
from .world_model_math_training import MathCaseEvaluation, MathTrainingCase, MathWorldModelEvaluationSummary, MathWorldModelTrainingSummary, WorldModelMathTrainer, ensure_starter_math_cases, load_math_training_cases
from .visual_geometry_3d import Scene3DReconstruction, ScenePrimitive3D, SceneRelation3D, VisualGeometry3DWorkbench, VisualGeometryBatchReconstructionSummary, VisualGeometryCollectionSummary, VisualGeometryDatasetSummary
from .vlso.real_image_eval import RealImageEvalBuilder, RealImageEvalCaseCandidate, RealImageEvalBuildSummary, RealImageEvalFinalizeSummary
from .vlso import AffordanceLabelDataset, AffordanceLabelExample, AffordanceLabelTarget, AffordancePrediction, AffordanceTrainingSummary, AffordanceWeightTrainer, DEFAULT_VISION_MODEL_ROOT, DetectorOutputAdapter, FrontierBundleItem, FrontierInstallSummary, FrontierSetupSummary, FrontierVisionInstaller, FrontierVisualSummary, FrontierVisionAdapter, FrontierVisionSpec, VisualSceneAdjudication, VisualSceneAdjudicator, GeometryPrimitiveBackbone, GeometryPrimitiveResult, GeometryReasoningResult, GeometryTopologyExtractor, GeometryTopologyResult, HybridMemoryMatch, ImageMaskPreprocessor, ImageParseResult, ImagePreprocessResult, JepaStructuralPredictor, PredictivePriorResult, LocalTextGenerator, OpenImagesAnnotationAdapter, OpenImagesPayloadSummary, OperatorType, RawImageObservationParser, SemanticRegionHypothesis, SemanticSceneAnalyzer, SemanticSceneHypothesis, SemanticSceneSummary, SharedWorldModel, StructuralOperatorBinding, SyntheticGeometryScene, SyntheticGeometrySceneBuilder, VisualAffordanceCandidate, VisualAffordanceFeatureExtractor, VisualClusterReviewDecision, VisualClusterReviewStore, VisualApprovedReviewRetrainer, VisualReviewRetrainSummary, VisualCollectionPlanItem, VisualCollectionRecord, VisualCollectionRunSummary, VisualCollectionSource, VisualConceptDataset, VisualConceptExample, VisualConceptLabelRecommender, VisualConceptLearningSummary, VisualConceptMatch, VisualConceptMemory, VisualConceptPrototypeTrainer, VisualConceptRecord, VisualConceptSelfTrainer, VisualConceptTarget, VisualDataCollector, VisualDownloadEntry, VisualDownloadSummary, VisualFamilyBatchSummary, VisualEmbeddingMatch, VisualEmbeddingRecord, VisualEmbeddingStore, VisualGeometryReasoner, VisualHybridMemory, VisualHybridMemoryResult, VisualObjectReasoner, VisualObjectReasoningResult, VisualObservation, VisualOperatorLearningSummary, VisualOperatorMatch, VisualOperatorMemory, VisualOperatorPrototypeTrainer, VisualOperatorRecord, VisualPseudoCluster, VisualSelfTrainingSummary, VisualStructuralOperatorInducer, VisualStructuralReasoningResult, VISION_BACKBONE_SPECS, VLSOAligner, VLSOAnswer, VLSOEntity, VLSOLanguageParser, VLSOOperator, VLSOQuestionAnswerer, VLSOReasoner, VLSORelation, VLSOVisualParser, VLSO_OPERATOR_TYPES, VisionBackboneSpec, VisionEmbeddingExtractor, WeakAffordanceClassifier, PseudoLabelAcceptanceConfig, GeometryPipelineSummary, VisualGeometryBootstrapPipeline, VlsoEvalCase, VlsoEvalResult, VlsoEvalSummary, VlsoGroundedEvaluator, VlsoReviewImpactEvaluator, VlsoStoreComparisonSummary, build_object_family_manifest, build_geometry_seed_manifest, resolve_local_vision_model_path

__all__ = [
    "AffordanceLabelDataset",
    "AffordanceLabelExample",
    "AffordanceLabelTarget",
    "AffordancePrediction",
    "AffordanceTrainingSummary",
    "AffordanceWeightTrainer",
    "ArithmeticReasoner",
    "AuditItem",
    "BASIS_OPERATOR_AXES",
    "basis_operator_axis",
    "canonicalize_basis_signature",
    "infer_basis_operators",
    "normalize_operator_symbol",
    "BASELINE_SPECS",
    "DEFAULT_VISION_MODEL_ROOT",
    "CURATED_PRESETS",
    "CURATED_PUBLIC_DATASETS",
    "CompetitiveProgrammingReasoner",
    "ClaimGrounding",
    "ContextFrame",
    "ContestProblemStructure",
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
    "CpParserComparisonSummary",
    "CpParserEvaluator",
    "CpParserPrediction",
    "CpParserTrainConfig",
    "CpLoraExperimentConfig",
    "CpLoraExperimentRunner",
    "CpLoraExperimentSummary",
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
    "CpLabeledDatasetDownloader",
    "CpLabeledDatasetEntry",
    "CpLabeledDatasetSummary",
    "CpGeometryEvalBuildSummary",
    "CpGeometryEvalBuilder",
    "CppCompileResult",
    "CppSyntaxChecker",
    "CopilotRequest",
    "CopilotResult",
    "SemOpCommonEvaluator",
    "SemOpEvalSnapshot",
    "CorpusBuilder",
    "AnalogyPolicyModel",
    "AnalogyPolicyScorer",
    "AnalogyPolicyTrainer",
    "AnalogyPolicyTrainingSummary",
    "CapabilityAuditRunner",
    "CapabilityAuditSummary",
    "CapabilityAxisStatus",
    "CapabilityImprovementRunner",
    "CapabilityImprovementSummary",
    "CapabilityCurriculumRound",
    "RTX4060CollectionLane",
    "ConceptFusionEngine",
    "ConceptFusionHypothesis",
    "ConceptFusionSummary",
    "UltimateAGIAxis",
    "UltimateAGIReadinessRunner",
    "UltimateAGIReadinessSummary",
    "RTX4060OptimizationSummary",
    "RTX4060ReasoningCoach",
    "UnifiedParserModel",
    "UnifiedParserTrainer",
    "UnifiedParserTrainingSummary",
    "LearnedUnifiedParser",
    "RetainedOperatorAlgebra",
    "RetainedOperatorModel",
    "RetainedOperatorRecord",
    "RetainedOperatorTrainer",
    "RetainedOperatorTrainingSummary",
    "OperatorRepairAttempt",
    "OperatorRepairEngine",
    "OperatorRepairPolicyModel",
    "OperatorRepairPolicyScorer",
    "OperatorRepairPolicyTrainer",
    "OperatorRepairPolicyTrainingSummary",
    "RetainedRepairProgramLibrary",
    "RetainedRepairProgramModel",
    "RetainedRepairProgramRecord",
    "RetainedRepairProgramTrainer",
    "RetainedRepairProgramTrainingSummary",
    "UnifiedSemOpArtifacts",
    "UnifiedSemOpTrainer",
    "UnifiedSemOpTrainingSummary",
    "UnifiedBenchmarkHarness",
    "UnifiedBenchmarkSummary",
    "GeneralizationGoalAxis",
    "GeneralizationGoalTracker",
    "GeneralizationProofEvidence",
    "GeneralizationProofHarness",
    "GeneralizationProofRound",
    "GeneralizationProofSummary",
    "GroundingReflection",
    "GroundingSelfEvolutionRound",
    "GroundingSelfEvolutionRunner",
    "GroundingSelfEvolutionSummary",
    "AnalogyEvalCase",
    "GroundedExplanationEvalCase",
    "CompilerRepairEvalCase",
    "OperatorContextAnalyzer",
    "ContinuousLearningBundleBuilder",
    "ContinuousLearningBundleSummary",
    "EnvironmentBrainRunner",
    "EnvironmentBrainSummary",
    "EnvironmentConceptStat",
    "EnvironmentRoutineStat",
    "EnvironmentProbe",
    "AdaptiveActionRehearsal",
    "AdaptiveActionStep",
    "AdaptiveEnvironmentAxis",
    "AdaptiveEnvironmentLearningRunner",
    "AdaptiveEnvironmentLearningSummary",
    "EvolvingImprovementProgram",
    "RecursiveSelfEvolutionRunner",
    "RecursiveSelfEvolutionSummary",
    "SelfEvolutionGenerationSummary",
    "GraphSupervisionExample",
    "GraphSupervisionExporter",
    "GraphSupervisionExportSummary",
    "CorpusMemoryStore",
    "CorpusReasoningLearner",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DOMAIN_PROFILES",
    "DOMAIN_TEMPLATES",
    "DomainCopilot",
    "DomainProfile",
    "DocumentEvidenceReasoner",
    "TeacherTraceExporter",
    "TeacherTraceRecord",
    "DistillationSftRecord",
    "DownloadedArtifact",
    "EmbeddingModelCache",
    "EmergentOperatorInducer",
    "FeedbackRule",
    "FeedbackRuleSet",
    "GeometryPrimitiveBackbone",
    "FrontierBundleItem",
    "FrontierInstallSummary",
    "FrontierSetupSummary",
    "FrontierVisionInstaller",
    "FrontierVisualSummary",
    "FrontierVisionAdapter",
    "FrontierVisionSpec",
    "VisualSceneAdjudication",
    "VisualSceneAdjudicator",
    "FunctorHypothesis",
    "OperatorAlgebraLearner",
    "OperatorAlgebraSummary",
    "OperatorAlgebraEvalCase",
    "OperatorAlgebraEvaluator",
    "OperatorAlgebraEvalSummary",
    "OperatorCurriculumBuilder",
    "OperatorCurriculumPhase",
    "OperatorLearningBundleSummary",
    "OperatorTransferEvaluator",
    "OperatorTransferEvalSummary",
    "OperatorTransferEvalCase",
    "OperatorSelfEvolutionEngine",
    "OperatorSelfEvolutionLoop",
    "OperatorProposalEngine",
    "OperatorProposalPattern",
    "OperatorProposalSummarizer",
    "ModelProposedOperator",
    "OperatorProposalComparator",
    "OperatorProposalComparisonSummary",
    "VisualSignalImpactEvaluator",
    "VisualSignalImpactSummary",
    "HybridOperatorProposalPolicy",
    "HybridOperatorProposalSummary",
    "OperatorEvolutionRunResult",
    "OperatorEvolutionSummary",
    "EvolvedOperatorProposal",
    "OperatorDecomposition",
    "OperatorTrainConfig",
    "OperatorTrainingScaffold",
    "ProgressAxis",
    "OperatorIntelligenceProgressEstimator",
    "OperatorIntelligenceProgress",
    "GeometryPrimitiveResult",
    "LocalTextGenerator",
    "SemanticRegionHypothesis",
    "SemanticSceneAnalyzer",
    "SemanticSceneHypothesis",
    "SemanticSceneSummary",
    "LogicalPatternMatcher",
    "LogicalPatternMatch",
    "VerificationCheck",
    "PatternOutcomeTrainer",
    "HardProblemReport",
    "LocalDependencyStatus",
    "LocalHardwareProfile",
    "detect_local_hardware",
    "detect_local_ml_stack",
    "recommended_generation_tokens",
    "should_force_4bit",
    "JepaStructuralPredictor",
    "PredictivePriorResult",
    "IntelligenceAxis",
    "IntelligenceSubsystem",
    "OperatorIntelligenceMap",
    "build_operator_intelligence_map",
    "HardProblemEngine",
    "HiddenPremiseEvalCase",
    "HiddenPremiseEvaluator",
    "HiddenPremiseEvalSummary",
    "HiddenPremiseExplorer",
    "HiddenPremiseResult",
    "HardProblemCandidate",
    "LabeledOpsCase",
    "LabeledOpsCaseResult",
    "LabeledOpsEvaluator",
    "LeWorldModelAdapter",
    "LeWorldModelAlignment",
    "LeWorldModelArtifact",
    "LeWorldModelSequenceRecord",
    "LeWorldModelPlan",
    "LeWorldModelPlannedAction",
    "LeWorldModelTrajectoryPlanner",
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
    "VisualSelfTrainingSummary",
    "VisualOperatorLearningSummary",
    "VisualOperatorMatch",
    "VisualOperatorMemory",
    "VisualOperatorPrototypeTrainer",
    "VisualOperatorRecord",
    "VisualPseudoCluster",
    "VisualStructuralOperatorInducer",
    "VisualStructuralReasoningResult",
    "VisualConceptSelfTrainer",
    "VisualClusterReviewDecision",
    "VisualClusterReviewStore",
    "VisualApprovedReviewRetrainer",
    "VisualReviewRetrainSummary",
    "StructuralOperatorBinding",
    "PseudoLabelAcceptanceConfig",
    "PublicCorpusIngestor",
    "PublicDatasetAdapter",
    "QueryPriorEffect",
    "MultimodalAlignmentMemory",
    "MultimodalAlignmentModel",
    "MultimodalAlignmentRecord",
    "MultimodalAlignmentTrainer",
    "MultimodalAlignmentTrainingSummary",
    "RagBaselineResult",
    "RegistryNode",
    "RemoteDatasetDownloader",
    "ResponseSynthesizer",
    "BeginnerOneClickSummary",
    "RuntimeDoctorReport",
    "RuntimeLaunchEntry",
    "RuntimeLaunchSummary",
    "RuntimeSurfaceSpec",
    "ReviewQueueItem",
    "ReviewQueueStore",
    "PromptUnderstandingAnalyzer",
    "PromptUnderstandingSummary",
    "FrameSituationSummary",
    "TemporalSceneReasoner",
    "TemporalSituationSummary",
    "infer_review_severity",
    "normalize_review_severity",
    "severity_weight",
    "RealImageEvalBuilder",
    "RealImageEvalCaseCandidate",
    "RealImageEvalBuildSummary",
    "RealImageEvalFinalizeSummary",
    "StructuredMeaningPipeline",
    "ScriptCompatibilityBreakdown",
    "ScriptCompatibilityModel",
    "ScriptCompatibilityScorer",
    "ScriptCompatibilityTrainer",
    "ScriptCompatibilityTrainingSummary",
    "GoalPreservationCheck",
    "PremiseCandidate",
    "PremiseValidation",
    "GraphFunctorHypothesis",
    "GraphOperatorDecomposition",
    "OperatorCompiler",
    "OperatorExecutor",
    "compile_and_execute",
    "OperatorInstruction",
    "OperatorExecutionReport",
    "SymbolicReasoner",
    "SymbolicResult",
    "SynthesizedResponse",
    "TextChunk",
    "TransferEvaluator",
    "QuantizedReviewAssignment",
    "TurboQuantReviewPlanner",
    "TurboQuantReviewSummary",
    "SemOpUnderstandingEvaluator",
    "UnderstandingEvalSummary",
    "MathStrategyPrior",
    "MathWorldCandidate",
    "MathWorldCheck",
    "WorldModelMathReasoner",
    "WorldModelMathReport",
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
    "VisualFamilyBatchSummary",
    "build_object_family_manifest",
    "build_geometry_seed_manifest",
    "VlsoEvalCase",
    "VlsoEvalResult",
    "VlsoEvalSummary",
    "VlsoGroundedEvaluator",
    "VlsoReviewImpactEvaluator",
    "VlsoStoreComparisonSummary",
    "VisualCollectionSource",
    "GeometryPipelineSummary",
    "VisualGeometryBootstrapPipeline",
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
    "VisualHybridMemory",
    "VisualHybridMemoryResult",
    "VisualObjectReasoner",
    "VisualObjectReasoningResult",
    "GeometryReasoningResult",
    "HybridMemoryMatch",
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
    "resolve_operating_policy",
    "resolve_review_severity_weight",
    "resolve_slice_balance_limit",
    "resolve_slice_thresholds",
    "DEFAULT_REVIEW_SEVERITY_WEIGHTS",
    "promoted_review_details",
    "collect_review_promotion_decisions",
    "evaluate_review_promotion",
    "infer_operating_domain",
    "DomainOperatingPolicy",
    "ReviewPromotionDecision",
    "OPERATING_POLICIES",
    "resolve_embedding_model_id",
    "build_runtime_doctor_report",
    "launch_beginner_one_click",
    "launch_runtime_stack",
    "review_reasons_from_kpis",
    "review_reasons_from_graph",
    "review_reasons_from_graph_and_kpis",
    "split_context_into_chunks",
    "train_repair_policy_from_graphs",
    "RepairUtilityModel",
    "RepairUtilityScorer",
    "RepairUtilityTrainer",
    "RepairUtilityTrainingSummary",
    "train_repair_utility_from_graphs",
    "train_retained_repair_programs_from_graphs",
    "train_multimodal_alignment_from_graphs",
    "export_graph_supervision_from_graphs",
    "build_continuous_learning_bundle",
    "BenchmarkGateDecision",
    "BenchmarkGateThresholds",
    "BenchmarkGatedContinuousTrainer",
    "BenchmarkGatedTrainingSummary",
    "BenchmarkSliceSummary",
    "PersistentBenchmarkCorpusSummary",
    "PromotedReviewBenchmarkCases",
]


































