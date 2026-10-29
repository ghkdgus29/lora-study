"""3단계: 분류의 loss - logits, softmax, cross-entropy, 그리고 라벨 불균형과 가중치.

train_lora.py의 answer_loss()가 쓰는 것:
    ce = F.cross_entropy(logits, targets, reduction="none")      # 토큰(=클래스)마다 -log P(정답)
    weight = where(targets == y_id, y_weight, 1.0)               # ' Y' 토큰에 가중치
    loss = (ce * weight).sum() / weight.sum()
LLM의 "다음 토큰 예측"은 어휘 248,320개 중 하나를 고르는 분류 문제다. 여기서는 클래스 3개, 2개로 같은 계산을 본다.

실행:  uv run steps/03_cross_entropy.py
       uv run steps/03_cross_entropy.py --y-weight 3
       uv run steps/03_cross_entropy.py --y-weight 9
"""

import argparse

import torch
import torch.nn.functional as F


def part_a_softmax_ce() -> None:
    print("=" * 70)
    print("A. logits -> softmax -> 확률 -> cross-entropy")
    logits = torch.tensor([2.0, 0.5, -1.0])  # 모델이 클래스 3개에 대해 낸 '점수'(아직 확률 아님)
    probs = torch.softmax(logits, dim=-1)  # exp(점수) / exp 합 -> 합이 1인 확률
    print("logits =", logits.tolist())
    print("softmax =", [round(p, 4) for p in probs.tolist()], " (합 =", round(probs.sum().item(), 4), ")")
    print("손으로: exp(2.0) / (exp(2.0)+exp(0.5)+exp(-1.0)) =", round((torch.exp(logits[0]) / torch.exp(logits).sum()).item(), 4))
    for target in (0, 2):
        ce_manual = -torch.log(probs[target])
        ce_torch = F.cross_entropy(logits[None], torch.tensor([target]))  # cross_entropy는 softmax + -log 를 한 번에 한다
        print(f"정답이 {target}번이면: P(정답) = {probs[target].item():.4f}, cross-entropy = -log P = {ce_manual.item():.4f}  (F.cross_entropy = {ce_torch.item():.4f})")
    print("P(정답)이 1에 가까우면 loss 0, 0에 가까우면 loss가 끝없이 커진다 -> '확신하며 틀리는 것'에 큰 벌점")
    print("train_lora.py 로그의 loss도 이 값이다: 답 토큰(52개)마다 -log P(정답 토큰)의 평균. 0.1이면 평균 P(정답) 약 0.9")


def part_b_gradient() -> None:
    print("\n" + "=" * 70)
    print("B. cross-entropy의 기울기 = (softmax 확률) - (정답 one-hot). 정답 점수는 올리고 나머지는 내린다")
    logits = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
    target = 2
    F.cross_entropy(logits[None], torch.tensor([target])).backward()
    probs = torch.softmax(logits.detach(), dim=-1)
    onehot = torch.zeros(3)
    onehot[target] = 1
    print("logits.grad     =", [round(g, 4) for g in logits.grad.tolist()])
    print("softmax - onehot=", [round(g, 4) for g in (probs - onehot).tolist()], " <- 같다")
    print("정답(2번)의 기울기만 음수 -> 기울기 하강이 2번 점수를 올리고 0번·1번 점수를 내린다. 이것이 '학습'이 하는 일의 전부")


def make_imbalanced(n: int = 2000, y_ratio: float = 0.1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n_y = int(n * y_ratio)
    y_pts = torch.randn(n_y, 2, generator=g) + torch.tensor([1.5, 1.5])  # Y: (1.5, 1.5) 주변
    n_pts = torch.randn(n - n_y, 2, generator=g)  # N: (0, 0) 주변 -> 겹치는 영역이 있다
    x = torch.cat([y_pts, n_pts])
    t = torch.cat([torch.ones(n_y, dtype=torch.long), torch.zeros(n - n_y, dtype=torch.long)])  # 1 = Y, 0 = N
    return x, t


def prf(pred: torch.Tensor, t: torch.Tensor) -> tuple[float, float, float]:
    tp = ((pred == 1) & (t == 1)).sum().item()
    p = tp / max(1, (pred == 1).sum().item())
    r = tp / max(1, (t == 1).sum().item())
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def part_c_imbalance(y_weight: float, steps: int) -> None:
    print("\n" + "=" * 70)
    print(f"C. 라벨 불균형: Y 10% / N 90%인 데이터를 선형 분류기로 학습. ' Y' 가중치 = {y_weight}")
    x, t = make_imbalanced()
    W = torch.zeros(2, 2, requires_grad=True)  # 입력 2차원 -> 클래스 2개 점수
    b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        logits = x @ W + b
        ce = F.cross_entropy(logits, t, reduction="none")  # 예시마다 -log P(정답)
        weight = torch.where(t == 1, torch.tensor(y_weight), torch.tensor(1.0))  # answer_loss()와 같은 식
        loss = (ce * weight).sum() / weight.sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        p_y = torch.softmax(x @ W + b, dim=-1)[:, 1]  # 예시마다 P(Y)
    print(f"  학습 후 평균 P(Y):  실제 Y인 예시에서 {p_y[t == 1].mean():.3f},  실제 N인 예시에서 {p_y[t == 0].mean():.3f}")
    print("  임계값별 (P(Y) >= 임계값이면 Y로 판정):")
    for th in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        p, r, f = prf((p_y >= th).long(), t)
        print(f"    임계값 {th:.1f}: Y로 판정 {(p_y >= th).float().mean() * 100:4.1f}%  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
    print("  가중치를 올리면 P(Y) 전체가 위로 밀려 같은 임계값에서 recall이 오르고 precision이 내린다. 임계값을 내려도 비슷한 효과.")
    print("  -> train_lora.py의 --y-weight와 eval_filter.py의 임계값 훑기는 같은 손잡이를 두 곳에서 돌리는 것이다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    part_a_softmax_ce()
    part_b_gradient()
    part_c_imbalance(args.y_weight, args.steps)


if __name__ == "__main__":
    main()
