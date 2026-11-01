"""2단계: 학습 루프의 뼈대 - optimizer, 배치, gradient accumulation, 학습률 스케줄, grad clipping.

실전 LLM 학습 루프를 파라미터 2개짜리(직선 맞추기) 문제로 그대로 재현한다.
    opt = torch.optim.AdamW(params, lr=...)
    sched = LambdaLR(opt, warmup + 선형 감소)
    for 예시:
        loss = ...
        (loss / accum).backward()
        if (i+1) % accum == 0:
            clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad()

실행:  uv run steps/02_training_loop.py
       uv run steps/02_training_loop.py --opt sgd --lr 0.1
       uv run steps/02_training_loop.py --opt adamw --lr 0.1
       uv run steps/02_training_loop.py --accum 4 --steps 30
       uv run steps/02_training_loop.py --by-hand       # md '손으로 계산해 보기'의 정답 (D)
"""

import argparse

import torch

TRUE_W, TRUE_B = 3.0, 1.0


def make_data(n: int = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g) * 6 - 3  # -3 ~ 3
    y = TRUE_W * x + TRUE_B + 0.3 * torch.randn(n, generator=g)  # 정답 직선 + 잡음
    return x, y


def mse(w, b, x, y) -> torch.Tensor:
    return ((w * x + b - y) ** 2).mean()


def make_scheduler(opt, total_steps: int, warmup_ratio: float):
    """실전과 같은 스케줄: warmup 동안 0 -> lr 로 올리고, 그 뒤 total_steps에서 0이 되게 직선으로 내린다."""
    warmup = max(1, int(total_steps * warmup_ratio))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))), warmup


def part_a_loop(args) -> None:
    print("=" * 70)
    print(f"A. 학습 루프: optimizer={args.opt}, lr={args.lr}, 예시 {args.batch}개씩, accum={args.accum}, optimizer step {args.steps}번")
    x, y = make_data()
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    params = [w, b]
    opt = torch.optim.SGD(params, lr=args.lr) if args.opt == "sgd" else torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched, warmup = make_scheduler(opt, args.steps, args.warmup_ratio)
    print(f"   학습률 스케줄: warmup {warmup} step 동안 0 -> {args.lr}, 그 뒤 step {args.steps}에서 0이 되게 선형 감소")

    g = torch.Generator().manual_seed(1)
    opt_step, micro = 0, 0
    while opt_step < args.steps:
        idx = torch.randint(0, len(x), (args.batch,), generator=g)  # 무작위로 batch개 뽑기 (실전은 순서를 섞어 하나씩)
        loss = mse(w, b, x[idx], y[idx])
        (loss / args.accum).backward()  # accum개의 평균 기울기가 되도록 미리 나눈다
        micro += 1
        if micro % args.accum == 0:
            # 기울기 벡터 길이가 clip을 넘으면 clip 길이로 줄인다. --clip 0이면 재기만 하고 자르지 않는다
            norm = torch.nn.utils.clip_grad_norm_(params, args.clip if args.clip > 0 else float("inf"))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step <= 3 or opt_step % 5 == 0 or opt_step == args.steps:
                print(f"  step {opt_step:3d}  loss {loss.item():8.4f}  w {w.item():7.3f}  b {b.item():7.3f}  grad_norm {norm.item():8.3f}  lr {sched.get_last_lr()[0]:.4f}")
    print(f"정답: w={TRUE_W}, b={TRUE_B}.  전체 데이터 loss = {mse(w, b, x, y).item():.4f} (잡음 때문에 0.09 근처가 바닥)")


def part_b_accumulation() -> None:
    print("\n" + "=" * 70)
    print("B. gradient accumulation: '32개 한 번'과 '16개씩 두 번 더하기'의 기울기는 같다")
    x, y = make_data()
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    mse(w, b, x[:32], y[:32]).backward()
    print(f"  32개 한 번:            w.grad = {w.grad.item():.6f}")
    w.grad, b.grad = None, None

    (mse(w, b, x[:16], y[:16]) / 2).backward()  # 첫 16개 - loss를 2로 나눠서 backward (grad가 '더해진다')
    (mse(w, b, x[16:32], y[16:32]) / 2).backward()  # 다음 16개
    print(f"  16개씩 두 번(각각 /2): w.grad = {w.grad.item():.6f}   <- 같다")
    print("  왜 쓰나: GPU 메모리에 한 번에 못 올리는 큰 배치를 작은 조각으로 나눠 같은 효과를 낸다.")
    print("  실전(긴 프롬프트)은 예시 하나(8k~14k 토큰)씩 forward하고 8개마다 step -> '배치 8'과 같은 기울기")


def part_c_clipping() -> None:
    print("\n" + "=" * 70)
    print("C. grad clipping: 기울기가 갑자기 커지면(이상한 예시, 긴 시퀀스) 한 step에 파라미터가 튀는 것을 막는다")
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x, y = make_data()
    mse(w, b, x * 50, y * 50).backward()  # 입력 크기를 키워 기울기를 일부러 크게
    before = torch.sqrt(w.grad**2 + b.grad**2).item()
    torch.nn.utils.clip_grad_norm_([w, b], max_norm=1.0)
    after = torch.sqrt(w.grad**2 + b.grad**2).item()
    print(f"  기울기 벡터 길이(norm): clip 전 {before:.1f} -> clip 후 {after:.3f}  (방향은 그대로, 길이만 1.0으로)")
    print("  실전: clip_grad_norm_(params, 1.0). 로그의 grad_norm이 자주 1을 크게 넘으면 lr을 낮추는 신호로 본다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=("sgd", "adamw"), default="adamw")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8, help="한 번 forward에 넣는 예시 수")
    ap.add_argument("--accum", type=int, default=1, help="몇 번 backward마다 optimizer step 할지")
    ap.add_argument("--steps", type=int, default=60, help="optimizer step 수")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=0.0, help="grad clipping 길이. 0이면 끔(실전은 1.0)")
    ap.add_argument("--by-hand", action="store_true", help="md '손으로 계산해 보기'의 정답을 torch로 찍는다")
    args = ap.parse_args()
    if args.by_hand:
        part_d_by_hand()
        return
    part_a_loop(args)
    part_b_accumulation()
    part_c_clipping()


def part_d_by_hand() -> None:
    """md '손으로 계산해 보기'와 같은 숫자. 예시 4개(y = 3x + 1 위의 점), w = b = 0에서 시작, loss = ½·mean((w·x + b − y)²)."""
    torch.set_printoptions(precision=4)
    x = torch.tensor([1.0, 2.0, -1.0, 0.0])
    y = torch.tensor([4.0, 7.0, -2.0, 1.0])

    def half_mse(w, b, xs, ys):
        return 0.5 * ((w * xs + b - ys) ** 2).mean()

    def grads(w, b, xs, ys):
        w = w.detach().clone().requires_grad_(True)
        b = b.detach().clone().requires_grad_(True)
        half_mse(w, b, xs, ys).backward()
        assert w.grad is not None and b.grad is not None
        return w.grad.item(), b.grad.item()

    print("=" * 70)
    print("D. 손계산 검산. 예시 (x, y) = (1,4) (2,7) (−1,−2) (0,1).  w = b = 0")
    w0, b0 = torch.tensor(0.0), torch.tensor(0.0)

    print("\n[1] 예시별 기울기 (배치 1)")
    for i in range(4):
        gw, gb = grads(w0, b0, x[i : i + 1], y[i : i + 1])
        print(f"  예시 {i + 1}: (x, y) = ({x[i].item():g}, {y[i].item():g})   dw = {gw:g}   db = {gb:g}")
    gw4, gb4 = grads(w0, b0, x, y)
    print(f"  배치 4 (평균):  dw = {gw4:g}   db = {gb4:g}")

    print("\n[2] accumulation: 2개씩 두 번, 각각 loss/2 로 backward 해서 더하기")
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    (half_mse(w, b, x[:2], y[:2]) / 2).backward()
    assert w.grad is not None and b.grad is not None
    print(f"  앞 2개 /2 뒤:   w.grad = {w.grad.item():g}   b.grad = {b.grad.item():g}")
    (half_mse(w, b, x[2:], y[2:]) / 2).backward()
    print(f"  뒤 2개 /2 더함: w.grad = {w.grad.item():g}   b.grad = {b.grad.item():g}   <- 배치 4와 같다")

    print("\n[3] clipping: 배치 4의 기울기 (dw, db)를 max_norm 1.0 으로")
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    half_mse(w, b, x, y).backward()
    norm = torch.nn.utils.clip_grad_norm_([w, b], 1.0)
    assert w.grad is not None and b.grad is not None
    print(f"  clip 전 norm = {norm.item():.4f}   clip 후 (dw, db) = ({w.grad.item():.4f}, {b.grad.item():.4f})   norm = {(w.grad**2 + b.grad**2).sqrt().item():.4f}")

    print("\n[4] SGD vs Adam, lr 0.1, step 2번 (clip 없음)")
    for name in ("sgd", "adam"):
        w = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.SGD([w, b], lr=0.1) if name == "sgd" else torch.optim.AdamW([w, b], lr=0.1, weight_decay=0.0)
        print(f"  {name}:")
        for step in (1, 2):
            loss = half_mse(w, b, x, y)
            loss.backward()
            assert w.grad is not None and b.grad is not None
            gw, gb = w.grad.item(), b.grad.item()
            opt.step()
            opt.zero_grad()
            extra = ""
            if name == "adam":
                st = opt.state[w]
                extra = f"   [w의 상태: m = {st['exp_avg'].item():.4f}, v = {st['exp_avg_sq'].item():.6f}]"
            print(f"    step {step}: loss {loss.item():.4f}  dw {gw:.4f}  db {gb:.4f}  ->  w {w.item():.4f}  b {b.item():.4f}{extra}")

    print("\n[5] 스케줄 LambdaLR: total 10, warmup 2. step마다 lr 배수")
    w = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.SGD([w], lr=1.0)
    sched, _ = make_scheduler(opt, 10, 0.2)
    mults = [sched.get_last_lr()[0]]
    for _ in range(10):
        opt.step()
        sched.step()
        mults.append(sched.get_last_lr()[0])
    print("  s     :", " ".join(f"{s:5d}" for s in range(11)))
    print("  배수  :", " ".join(f"{m:5.3f}" for m in mults))
    print("  <- s = 0 일 때 배수 0: warmup이 있으면 첫 optimizer step은 lr 0이라 파라미터가 안 움직인다")


if __name__ == "__main__":
    main()
