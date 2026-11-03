from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    """全局训练与运行设置。

    网络尺寸与论文一致；优化器、混合比例和每步训练预算是复现方案中的实现选择，
    不是作者已公布的超参数。
    """

    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    learning_rate: float = 1e-3
    local_learning_rate: float | None = None
    batch_size: int = 256
    train_steps_per_update: int = 1000
    uniform_mix_eta: float = 0.1
    gradient_clip_norm: float = 1.0
    seed: int = 42
    log_every: int = 10
    eval_batch_size: int = 2048
    exact_max_qubits: int = 16
    save_state: bool = False
    device: str | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    paper: str = "arXiv:2606.08707v1"
    paper_checked: str = "2026-10-21"
    implementation_choices: str = (
        "Pre-LN/GELU/末端 LN；每个更新重置 Adam；混合 spectator 采样；"
        "非对角门逐个更新；同层连续对角门合并；"
        "默认只保存配置/变更/结果，不保存权重"
    )

    def model_config(self, n_qubits: int) -> dict:
        return {
            "n_qubits": n_qubits,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TuningConfig:
    """可选的 Optuna 搜索，只调少数全局训练超参数。"""

    n_trials: int = 8
    timeout: float | None = None
    seed: int = 42
    search_learning_rate: bool = True
    search_eta: bool = True
    search_steps: bool = False
    lr_low: float = 3e-5
    lr_high: float = 3e-3
    eta_low: float = 0.0
    eta_high: float = 0.3
    steps_choices: tuple[int, ...] = field(default_factory=lambda: (100, 300, 1000))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["steps_choices"] = list(self.steps_choices)
        return data
