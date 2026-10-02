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
    "iq3_xxs": ModelVariant(
        key="iq3_xxs",
        repo="bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF",
        filename="Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf",
        nominal_gb=31.73,
        quality="medium",
        recommended_mode="native-full-residency",   # dev31: >= 40 GiB Macs
    ),
    # dev42: Qwen3-Coder-Next has the Qwen3-Next-80B-A3B architecture (rope_theta 5e6 instead of 1e7, read from the GGUF)
    "coder_iq2_xxs": ModelVariant(
        key="coder_iq2_xxs",
        repo="bartowski/Qwen_Qwen3-Coder-Next-GGUF",
        filename="Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf",
        nominal_gb=19.30,
        quality="very-low",
        recommended_mode="native-bounded-cache",
    ),
    "coder_iq3_xxs": ModelVariant(
        key="coder_iq3_xxs",
        repo="bartowski/Qwen_Qwen3-Coder-Next-GGUF",
        filename="Qwen_Qwen3-Coder-Next-IQ3_XXS.gguf",
        nominal_gb=31.73,
        quality="medium",
        recommended_mode="native-full-residency",
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
    "48gb": "iq3_xxs",
    "coder": "coder_iq2_xxs",
    "coder-48gb": "coder_iq3_xxs",
}

def resolve_variant(name: str) -> ModelVariant:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in VARIANTS:
        choices = ", ".join(sorted(set(VARIANTS) | set(ALIASES)))
        raise ValueError(f"Unknown variant '{name}'. Choices: {choices}")
    return VARIANTS[key]
