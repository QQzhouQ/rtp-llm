"""Ascend MoE placeholder: rejects all MoE configurations.

With the CANN MoE pipeline (AscendCannStrategy) registered ahead of this
placeholder, this fallback only stays as the last-resort guard for configs
the CANN strategy cannot handle (e.g. quantized MoE). It must reject
quietly (can_handle -> False) instead of raising: the strategy registry
evaluates every candidate inside a list comprehension, and an unconditional
raise here would crash strategy selection even when an earlier strategy
already matched.
"""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class AscendBf16FallbackStrategy(MoeStrategy):
    """Placeholder that rejects all MoE configurations on Ascend."""

    _REJECT_MSG = (
        "Ascend MoE has no strategy for this configuration (the CANN bf16 "
        "pipeline rejected it; no other NPU-capable MoE executor exists)."
    )

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        # Never selected when the CANN strategy accepts the config; reaching
        # here means no Ascend MoE implementation covers this configuration.
        # Record the rejection via the checker (no raise: the registry scans
        # candidates inside a list comprehension).
        checker.check(False, reason=cls._REJECT_MSG)

    def get_attributes(self) -> StrategyAttributes:
        # can_handle() consults attributes before check_conditions; a raise
        # here would escape the registry's candidate scan (it only catches
        # ImportError). Return a benign marker instead and let
        # check_conditions reject the config.
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.routers import (
            NpuPureTpRouter,
        )

        return StrategyAttributes(
            router_class=NpuPureTpRouter,
            executor_class=None,
            quant_config=None,
        )
