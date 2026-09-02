from dataclasses import dataclass

@dataclass(frozen=True)
class ModelVariant:
    key: str
    repo: str
    filename: str
    nominal_gb: float
    quality: str
    recommended_mode: str

MODEL_ID = "qwen3-next-80b-a3b-instruct"

VARIANTS = {
    "iq2_xxs": ModelVariant(
        key="iq2_xxs",
        repo="bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF",
        filename="Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf",
        nominal_gb=19.30,
        quality="very-low",
        recommended_mode="metal-resident",
    ),
    "iq2_xs": ModelVariant(
        key="iq2_xs",
        repo="bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF",
        filename="Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XS.gguf",
        nominal_gb=22.22,
        quality="low",
        recommended_mode="ssd-cpu",
    ),
    "iq2_s": ModelVariant(
        key="iq2_s",
        repo="bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF",
        filename="Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_S.gguf",
        nominal_gb=23.37,
        quality="low",
        recommended_mode="ssd-cpu",
    ),
    "q4_k_m": ModelVariant(
        key="q4_k_m",
        repo="Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF",
        filename="Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
        nominal_gb=48.40,
        quality="recommended-quality",
        recommended_mode="ssd-cpu",
    ),
}

ALIASES = {
    "24gb": "iq2_xxs",
    "small": "iq2_xxs",
    "balanced": "iq2_xs",
    "quality": "q4_k_m",
    "q4": "q4_k_m",
}

def resolve_variant(name: str) -> ModelVariant:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in VARIANTS:
        choices = ", ".join(sorted(set(VARIANTS) | set(ALIASES)))
        raise ValueError(f"Unknown variant '{name}'. Choices: {choices}")
    return VARIANTS[key]
