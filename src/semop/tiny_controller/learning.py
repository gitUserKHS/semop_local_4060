from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from semop.kernel.engine import ActionPolicy
from semop.kernel.self_learning import PolicyCandidate
from semop.kernel.traces import DecisionTrainingCase

from .numpy_runtime import NumpyTinyController, TinyControllerConfig


@dataclass(frozen=True)
class TinyControllerPolicyLearner:
    """Optional PyTorch trainer implementing the verifier-gated learner contract."""

    config: TinyControllerConfig = field(default_factory=TinyControllerConfig)
    epochs: int = 1
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 0

    name = "tiny-controller-v5"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("tiny-controller learner epochs must be positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("tiny-controller learning rate must be finite and positive")
        if self.max_grad_norm <= 0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("tiny-controller gradient norm must be finite and positive")
        if not self.device.strip():
            raise ValueError("tiny-controller learner device must not be empty")

    def train(
        self,
        cases: Sequence[DecisionTrainingCase],
        incumbent: ActionPolicy | None = None,
    ) -> PolicyCandidate:
        if not cases:
            raise ValueError("tiny-controller training requires decision cases")
        try:
            import torch

            from .training import (
                TorchTinyController,
                encode_training_example,
                train_controller,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TinyControllerPolicyLearner requires the optional train profile "
                "(NumPy and PyTorch)"
            ) from exc

        torch.manual_seed(self.seed)
        if isinstance(incumbent, NumpyTinyController):
            if incumbent.config != self.config:
                raise ValueError(
                    "incumbent tiny-controller config differs from learner config"
                )
            runtime = incumbent
        else:
            runtime = NumpyTinyController.random(self.config, seed=self.seed)
        model = TorchTinyController.from_numpy(runtime)
        max_step = max(case.step for case in cases)
        encoded = []
        terminal_examples = 0
        for case in cases:
            encoded.append(
                encode_training_example(
                    case.state,
                    case.goals,
                    case.actions,
                    target_action=case.target_action,
                    target_halt=False,
                    target_value=max(
                        0.0,
                        1.0 - (case.step - 1) / max(1, max_step),
                    ),
                    config=self.config,
                )
            )
            if case.terminal_state is not None:
                encoded.append(
                    encode_training_example(
                        case.terminal_state,
                        case.terminal_goals,
                        (),
                        target_action=-1,
                        target_halt=True,
                        target_value=1.0,
                        config=self.config,
                    )
                )
                terminal_examples += 1
        history = train_controller(
            model,
            encoded,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            device=self.device,
        )
        weights = {
            name: parameter.detach().cpu().numpy()
            for name, parameter in model.named_parameters()
        }
        learned = NumpyTinyController(self.config, weights)
        artifact = learned.to_artifact()
        return PolicyCandidate(
            policy=learned,
            kind=self.name,
            artifact_suffix=".npz",
            artifact=artifact,
            parameter_count=learned.parameter_count,
            training_updates=len(encoded) * self.epochs,
            diagnostics=(
                f"verified_decision_cases={len(cases)}",
                f"verified_terminal_cases={terminal_examples}",
                f"epochs={self.epochs}",
                "loss_history=" + ",".join(f"{value:.6f}" for value in history),
            ),
        )

    def restore(self, artifact: bytes) -> NumpyTinyController:
        restored = NumpyTinyController.from_artifact(artifact)
        if restored.config != self.config:
            raise ValueError(
                "restored tiny-controller config differs from learner config"
            )
        return restored
