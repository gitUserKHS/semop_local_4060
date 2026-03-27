from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LocalDependencyStatus:
    torch_available: bool = False
    torch_version: str = ''
    transformers_available: bool = False
    transformers_version: str = ''
    peft_available: bool = False
    peft_version: str = ''
    bitsandbytes_available: bool = False
    bitsandbytes_version: str = ''
    accelerate_available: bool = False
    accelerate_version: str = ''
    llm_ready: bool = False
    training_ready: bool = False
    qlora_ready: bool = False
    missing_core: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalHardwareProfile:
    requested_profile: str = 'auto'
    detected_profile: str = 'cpu_only'
    device: str = 'cpu'
    cuda_available: bool = False
    gpu_name: str = ''
    vram_gb: float = 0.0
    supports_fp16: bool = False
    supports_bf16: bool = False
    bitsandbytes_available: bool = False
    low_vram: bool = False
    recommended_precision: str = 'fp32'
    recommended_use_lora: bool = False
    recommended_use_qlora: bool = False
    recommended_batch_size: int = 1
    recommended_gradient_accumulation: int = 4
    recommended_generation_tokens: int = 256
    operator_algebra_mode: str = 'symbolic_cpu_first'
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)




def _package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return ''
    try:
        return str(version(name))
    except PackageNotFoundError:
        return ''
    except Exception:
        return ''

def _env_flag(name: str) -> bool | None:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return None
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return None


def _bitsandbytes_available() -> bool:
    return _package_available('bitsandbytes')


def detect_local_hardware(preferred_profile: str = 'auto') -> LocalHardwareProfile:
    forced_cuda = _env_flag('SEMOP_FORCE_CUDA')
    forced_name = os.getenv('SEMOP_FORCE_GPU_NAME', '').strip()
    forced_vram = os.getenv('SEMOP_FORCE_VRAM_GB', '').strip()
    gpu_name = forced_name
    vram_gb = 0.0
    cuda_available = False
    supports_fp16 = False
    supports_bf16 = False
    device = 'cpu'
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        if forced_cuda is not None:
            cuda_available = forced_cuda
        if cuda_available:
            device = 'cuda'
            if not gpu_name:
                try:
                    gpu_name = str(torch.cuda.get_device_name(0))
                except Exception:
                    gpu_name = 'CUDA GPU'
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(float(props.total_memory) / float(1024 ** 3), 2)
            except Exception:
                vram_gb = 0.0
            supports_fp16 = True
            try:
                supports_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                supports_bf16 = False
    if forced_vram:
        try:
            vram_gb = float(forced_vram)
        except ValueError:
            pass
    bnb = _bitsandbytes_available()
    normalized_name = gpu_name.lower()
    profile_name = 'cpu_only'
    low_vram = False
    notes: list[str] = []
    if preferred_profile and preferred_profile != 'auto':
        profile_name = preferred_profile.strip().lower()
        if profile_name == 'rtx_4060_8gb':
            cuda_available = True
            device = 'cuda'
            gpu_name = gpu_name or 'NVIDIA GeForce RTX 4060'
            if vram_gb <= 0.0:
                vram_gb = 8.0
            supports_fp16 = True
            low_vram = True
    elif cuda_available:
        if '4060' in normalized_name and (vram_gb <= 8.6 or vram_gb == 0.0):
            profile_name = 'rtx_4060_8gb'
            low_vram = True
        elif vram_gb and vram_gb <= 9.5:
            profile_name = 'cuda_low_vram'
            low_vram = True
        else:
            profile_name = 'cuda_general'
    if profile_name == 'rtx_4060_8gb':
        notes = [
            'RTX 4060 8GB profile: prefer LoRA/QLoRA, keep per-device batch size at 1, and keep the operator algebra path symbolic-first.',
            'Use 4-bit quantization when bitsandbytes is available; otherwise use fp16 with short generation lengths.',
        ]
        return LocalHardwareProfile(
            requested_profile=preferred_profile,
            detected_profile=profile_name,
            device='cuda',
            cuda_available=True,
            gpu_name=gpu_name or 'NVIDIA GeForce RTX 4060',
            vram_gb=vram_gb or 8.0,
            supports_fp16=True,
            supports_bf16=supports_bf16,
            bitsandbytes_available=bnb,
            low_vram=True,
            recommended_precision='fp16',
            recommended_use_lora=True,
            recommended_use_qlora=bnb,
            recommended_batch_size=1,
            recommended_gradient_accumulation=16,
            recommended_generation_tokens=384,
            operator_algebra_mode='symbolic_first_gpu_assist',
            notes=notes,
        )
    if profile_name == 'cuda_low_vram':
        notes = [
            'Low-VRAM CUDA profile: keep generation short and prefer parameter-efficient fine-tuning.',
            'Operator algebra, retrieval, and symbolic verifiers should run before any local LLM fallback.',
        ]
        return LocalHardwareProfile(
            requested_profile=preferred_profile,
            detected_profile=profile_name,
            device='cuda',
            cuda_available=True,
            gpu_name=gpu_name or 'CUDA GPU',
            vram_gb=vram_gb,
            supports_fp16=supports_fp16 or True,
            supports_bf16=supports_bf16,
            bitsandbytes_available=bnb,
            low_vram=True,
            recommended_precision='fp16',
            recommended_use_lora=True,
            recommended_use_qlora=bnb,
            recommended_batch_size=1,
            recommended_gradient_accumulation=12,
            recommended_generation_tokens=384,
            operator_algebra_mode='symbolic_first_gpu_assist',
            notes=notes,
        )
    if profile_name == 'cuda_general':
        notes = [
            'CUDA profile: GPU is available, but symbolic operator algebra should still remain the first stage for reliability.',
        ]
        return LocalHardwareProfile(
            requested_profile=preferred_profile,
            detected_profile=profile_name,
            device='cuda',
            cuda_available=True,
            gpu_name=gpu_name or 'CUDA GPU',
            vram_gb=vram_gb,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            bitsandbytes_available=bnb,
            low_vram=bool(vram_gb and vram_gb <= 12.0),
            recommended_precision='bf16' if supports_bf16 else 'fp16',
            recommended_use_lora=True,
            recommended_use_qlora=bool(bnb and vram_gb and vram_gb <= 12.0),
            recommended_batch_size=1,
            recommended_gradient_accumulation=8,
            recommended_generation_tokens=512,
            operator_algebra_mode='symbolic_first_gpu_assist',
            notes=notes,
        )
    return LocalHardwareProfile(
        requested_profile=preferred_profile,
        detected_profile='cpu_only',
        device=device,
        cuda_available=False,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        supports_fp16=False,
        supports_bf16=False,
        bitsandbytes_available=bnb,
        low_vram=False,
        recommended_precision='fp32',
        recommended_use_lora=False,
        recommended_use_qlora=False,
        recommended_batch_size=1,
        recommended_gradient_accumulation=4,
        recommended_generation_tokens=256,
        operator_algebra_mode='symbolic_cpu_first',
        notes=[
            'CPU-only profile: keep local neural components small and let operator algebra, retrieval, and rule-based verifiers carry most of the workload.',
        ],
    )


def recommended_generation_tokens(profile: LocalHardwareProfile, requested_tokens: int) -> int:
    return max(64, min(int(requested_tokens), int(profile.recommended_generation_tokens)))


def should_force_4bit(profile: LocalHardwareProfile, requested_use_4bit: bool) -> bool:
    if profile.detected_profile in {'rtx_4060_8gb', 'cuda_low_vram'}:
        return bool(profile.bitsandbytes_available and (requested_use_4bit or profile.recommended_use_qlora))
    return bool(requested_use_4bit and profile.bitsandbytes_available)



def detect_local_ml_stack(profile: LocalHardwareProfile | None = None) -> LocalDependencyStatus:
    resolved_profile = profile or detect_local_hardware()
    torch_available = _package_available('torch')
    transformers_available = _package_available('transformers')
    peft_available = _package_available('peft')
    bitsandbytes_available = _package_available('bitsandbytes')
    accelerate_available = _package_available('accelerate')
    missing_core: list[str] = []
    missing_optional: list[str] = []
    notes: list[str] = []
    if not torch_available:
        missing_core.append('torch')
    if not transformers_available:
        missing_core.append('transformers')
    if not peft_available:
        missing_optional.append('peft')
    if not accelerate_available:
        missing_optional.append('accelerate')
    if not bitsandbytes_available:
        missing_optional.append('bitsandbytes')
    llm_ready = torch_available and transformers_available
    training_ready = llm_ready and peft_available
    qlora_ready = bool(training_ready and accelerate_available and bitsandbytes_available and resolved_profile.cuda_available)
    if resolved_profile.detected_profile in {'rtx_4060_8gb', 'cuda_low_vram'}:
        if not bitsandbytes_available:
            notes.append('4-bit QLoRA is unavailable because bitsandbytes is missing, so low-VRAM CUDA will fall back to fp16 and shorter generations.')
        if not peft_available:
            notes.append('LoRA adapters are unavailable because peft is missing, so local fine-tuning will stay in dry-run or full-model mode only.')
        if llm_ready and not qlora_ready:
            notes.append('The 4060 profile will still run in symbolic/operator-algebra-first mode even without full QLoRA support.')
    if llm_ready:
        notes.append('Local LLM-assisted extraction is available.')
    else:
        notes.append('Local LLM-assisted extraction is blocked until torch and transformers are installed.')
    if training_ready:
        notes.append('Parameter-efficient fine-tuning is available.')
    else:
        notes.append('Fine-tuning is blocked until peft is installed alongside torch and transformers.')
    return LocalDependencyStatus(
        torch_available=torch_available,
        torch_version=_package_version('torch'),
        transformers_available=transformers_available,
        transformers_version=_package_version('transformers'),
        peft_available=peft_available,
        peft_version=_package_version('peft'),
        bitsandbytes_available=bitsandbytes_available,
        bitsandbytes_version=_package_version('bitsandbytes'),
        accelerate_available=accelerate_available,
        accelerate_version=_package_version('accelerate'),
        llm_ready=llm_ready,
        training_ready=training_ready,
        qlora_ready=qlora_ready,
        missing_core=missing_core,
        missing_optional=missing_optional,
        notes=notes,
    )
