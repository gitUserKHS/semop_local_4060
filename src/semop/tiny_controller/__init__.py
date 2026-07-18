"""Small learned action-ranking controller.

Only NumPy is required for inference. Import ``semop.tiny_controller.training``
explicitly when PyTorch training support is needed.
"""

from .features import (
    CanonicalAction,
    CanonicalProblemGraph,
    CanonicalRelation,
    ControllerFeatureProfile,
    canonicalize_problem,
    stable_bucket,
)
from .numpy_runtime import NumpyTinyController, TinyControllerConfig
from .linear_policy import StructuralLinearPolicy, StructuralPolicyLearner
from .learning import TinyControllerPolicyLearner
from .grounding_features import (
    GROUNDING_FEATURE_VERSION,
    GroundingFeatureVector,
    encode_grounding_candidate,
    grounding_feature_support,
)
from .grounding_curriculum import (
    DomainGroundingCurriculumAudit,
    GroundingCurriculumAudit,
    GroundingCurriculumSelection,
    audit_grounding_curriculum_selection,
    select_feature_novel_grounding_examples,
)
from .grounding_policy import (
    GroundingPolicyOutcome,
    GroundingPolicyReport,
    GroundingReviewItem,
    GroundingPrediction,
    SparseGroundingPolicy,
    select_grounding_review_candidates,
)
from .grounding_learning import (
    DomainGroundingMetrics,
    GroundingLearningBudget,
    GroundingLearningResult,
    GroundingOnlineLearningResult,
    GroundingOnlineState,
    GroundingPolicyCandidate,
    GroundingPolicyMetrics,
    GroundingPolicyStore,
    GroundingReplayBuffer,
    GroundingReplayItem,
    GroundingRiskCoveragePoint,
    GroundingSplitAudit,
    SparseGroundingLearner,
    VerifiedGroundingLearningLoop,
    VerifiedGroundingOnlineLearningLoop,
    audit_grounding_split,
    collect_grounding_examples,
    evaluate_grounding_policy,
    grounding_risk_coverage_curve,
)

__all__ = [
    "CanonicalAction",
    "CanonicalProblemGraph",
    "CanonicalRelation",
    "ControllerFeatureProfile",
    "DomainGroundingMetrics",
    "GROUNDING_FEATURE_VERSION",
    "GroundingFeatureVector",
    "DomainGroundingCurriculumAudit",
    "GroundingCurriculumAudit",
    "GroundingCurriculumSelection",
    "GroundingLearningBudget",
    "GroundingLearningResult",
    "GroundingOnlineLearningResult",
    "GroundingOnlineState",
    "GroundingPolicyCandidate",
    "GroundingPolicyMetrics",
    "GroundingPolicyOutcome",
    "GroundingPolicyReport",
    "GroundingReviewItem",
    "GroundingPolicyStore",
    "GroundingPrediction",
    "GroundingReplayBuffer",
    "GroundingReplayItem",
    "GroundingRiskCoveragePoint",
    "GroundingSplitAudit",
    "NumpyTinyController",
    "SparseGroundingLearner",
    "SparseGroundingPolicy",
    "StructuralLinearPolicy",
    "StructuralPolicyLearner",
    "TinyControllerConfig",
    "TinyControllerPolicyLearner",
    "VerifiedGroundingLearningLoop",
    "VerifiedGroundingOnlineLearningLoop",
    "audit_grounding_split",
    "canonicalize_problem",
    "collect_grounding_examples",
    "encode_grounding_candidate",
    "grounding_feature_support",
    "evaluate_grounding_policy",
    "grounding_risk_coverage_curve",
    "audit_grounding_curriculum_selection",
    "select_grounding_review_candidates",
    "select_feature_novel_grounding_examples",
    "stable_bucket",
]
