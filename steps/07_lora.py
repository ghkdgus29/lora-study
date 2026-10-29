"""7단계: LoRA - 큰 가중치는 얼리고, 작은 '보정 행렬' 두 개만 학습한다.

train_lora.py가 쓰는 것:
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=[...q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj...])
    model = get_peft_model(model, lora); model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]      # LoRA 파라미터만 optimizer에
    model.save_pretrained(out)                                        # 어댑터(작은 파일)만 저장

실행:  uv run steps/07_lora.py
       uv run steps/07_lora.py --r 4            # r을 바꾸면 파라미터 수가 어떻게 변하나
       uv run steps/07_lora.py --r 64
       uv run steps/07_lora.py --targets q_proj,v_proj   # attention 일부에만
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ALL_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  # train_lora.py의 LORA_TARGET_MODULES 중 0.6B에 있는 것


class LoRALinear(nn.Module):
    """nn.Linear 하나에 LoRA를 손으로 붙인 것. 출력 = W x + (B (A x)) * (alpha / r).  W는 고정, A·B만 학습."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)  # 원래 가중치는 얼린다
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)  # 입력 -> r차원 (작은 난수)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # r차원 -> 출력 (0으로 시작!)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def part_a_manual(r: int, alpha: int) -> None:
    print("=" * 70)
    print(f"A. 손으로 만든 LoRA: nn.Linear(256 -> 256) 에 r={r}, alpha={alpha}")
    torch.manual_seed(0)
    base = nn.Linear(256, 256)
    lora = LoRALinear(base, r, alpha)
    x = torch.randn(4, 256)
    print(f"  원래 가중치 W: {tuple(base.weight.shape)} = {base.weight.numel():,}개 (얼림)")
    print(f"  LoRA A: {tuple(lora.A.shape)} + B: {tuple(lora.B.shape)} = {lora.A.numel() + lora.B.numel():,}개 (학습) = W의 {(lora.A.numel() + lora.B.numel()) / base.weight.numel() * 100:.1f}%")
    print(f"  학습 전 출력이 원래와 같은가: {torch.allclose(lora(x), base(x))}   <- B가 0이라 보정이 0. '처음에는 원래 모델 그대로'에서 출발")
    for rr in (4, 16, 64, 256):
        print(f"    r={rr:3d}: A+B = {rr * (256 + 256):6,}개 ({rr * 512 / 65536 * 100:5.1f}%)  <- r에 비례. r=256이면 W와 같은 크기(의미 없음)")
    # 한 step 학습해 보면 출력이 달라진다
    target = torch.randn(4, 256)
    opt = torch.optim.Adam([lora.A, lora.B], lr=0.1)
    for _ in range(5):
        loss = ((lora(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"  5 step 뒤 출력이 원래와 같은가: {torch.allclose(lora(x), base(x))}   (B가 0에서 벗어나 보정이 생겼다). W는 그대로: {base.weight.grad is None}")
    print(f"  scale = alpha/r = {alpha}/{r} = {alpha / r}: 보정의 세기. 관례는 alpha = 2r (train_lora.py: r 16, alpha 32 -> 2.0)")


def part_b_peft(model_name: str, r: int, alpha: int, targets: list[str], device: str):
    print("\n" + "=" * 70)
    print(f"B. PEFT로 실제 모델에 붙이기: {model_name}, r={r}, alpha={alpha}, target_modules={targets}")
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    layer0 = model.model.layers[0]
    print("  층 하나(layers[0])의 nn.Linear 목록 - target_modules는 이 이름의 끝부분과 맞는 것에 붙는다:")
    for name, mod in layer0.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    {name:22s} {tuple(mod.weight.shape)}")
    print(f"  (Qwen3.6-35B에는 여기에 Gated DeltaNet의 in_proj_qkv/in_proj_z/out_proj, 공유 expert, 그리고 nn.Linear가 아닌 256개 expert 3차원 파라미터가 더 있다)")

    ids = torch.tensor(tok.encode("[1] Y\n[2]", add_special_tokens=False), device=device)[None]
    with torch.no_grad():
        before = model(input_ids=ids).logits[0, -1].float()

    total = sum(p.numel() for p in model.parameters())
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    q = model.base_model.model.model.layers[0].self_attn.q_proj
    print(f"  감싼 뒤 layers[0].self_attn.q_proj 의 타입: {type(q).__name__}  (안에 lora_A {tuple(q.lora_A['default'].weight.shape)}, lora_B {tuple(q.lora_B['default'].weight.shape)})")
    with torch.no_grad():
        after = model(input_ids=ids).logits[0, -1].float()
    print(f"  감싼 직후 출력이 원래와 같은가: {torch.allclose(before, after)}   <- A와 같은 이유(B=0)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  메모리 어림(fp32 기울기 + Adam 상태 2개 = 파라미터당 12바이트):")
    print(f"    전체 파인튜닝: {total:,}개 x 12B = {total * 12 / 1e9:6.2f}GB  (35B면 {35e9 * 12 / 1e9:.0f}GB - 96GB GPU 한 장에 못 넣는다)")
    print(f"    LoRA:          {trainable:,}개 x 12B = {trainable * 12 / 1e9:6.2f}GB")
    return model, tok


def part_c_save(model) -> None:
    print("\n" + "=" * 70)
    print("C. save_pretrained: 어댑터(A, B)만 저장한다. 원래 가중치는 저장하지 않는다")
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        for f in sorted(Path(d).iterdir()):
            print(f"  {f.name:28s} {f.stat().st_size / 1e6:7.2f}MB")
    print("  서비스(vLLM)는 원래 모델 + 이 어댑터 폴더를 같이 읽는다. 학습 결과물 = 이 폴더 하나")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default=ALL_TARGETS, help="쉼표로 구분")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    part_a_manual(args.r, args.alpha)
    model, _ = part_b_peft(args.model, args.r, args.alpha, args.targets.split(","), device)
    part_c_save(model)


if __name__ == "__main__":
    main()
