"""1단계: 텐서와 자동 미분(autograd), 그리고 "기울기를 따라 내려가기".

모든 학습 루프에 있는 이 세 줄이 무엇을 하는지 가장 작은 예로 본다.
    loss.backward()            # 기울기 계산
    opt.step()                 # 기울기 방향으로 파라미터 수정
    opt.zero_grad()            # 기울기 비우기

실행:  uv run steps/01_tensor_autograd.py
       uv run steps/01_tensor_autograd.py --lr 0.3      # 학습률이 너무 크면?
       uv run steps/01_tensor_autograd.py --lr 0.01     # 너무 작으면?
       uv run steps/01_tensor_autograd.py --by-hand     # md '손으로 계산해 보기'의 정답 (E)
       uv run steps/01_tensor_autograd.py --by-hand2    # md '손으로 계산해 보기 2'의 정답 (F): 층 2개, step 2번
       uv run steps/01_tensor_autograd.py --by-hand3    # md '손으로 계산해 보기 3'의 정답 (G): bias + ReLU
"""

import argparse

import torch


def part_a_tensor() -> None:
    print("=" * 70)
    print("A. 텐서 = 숫자 배열. 모델의 파라미터도, 입력 토큰도, loss도 전부 텐서다")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("x =", x)
    print("x.shape =", tuple(x.shape), " (2행 3열)   x.dtype =", x.dtype)
    print("x * 2 =", (x * 2).tolist())
    print("x.sum() =", x.sum().item(), "  x.mean() =", round(x.mean().item(), 3))
    # [None]: 맨 앞에 길이 1짜리 차원 추가. 토큰 id 리스트를 "문장 1개짜리 배치"로 만들 때 쓴다
    ids = torch.tensor([5, 12, 7])
    print("ids.shape =", tuple(ids.shape), " ids[None].shape =", tuple(ids[None].shape), "  <- [L] -> [1, L] (배치 차원 추가)")
    # bf16: LLM 가중치를 올릴 때 흔히 쓰는 형식. 자릿수가 적어 메모리가 절반(fp32 4바이트 -> 2바이트)
    y = torch.tensor([1 / 3], dtype=torch.float32)
    print("1/3 을 fp32로:", y.item(), "  bf16으로:", y.to(torch.bfloat16).item(), "  <- bf16은 정밀도가 낮다(메모리 절반)")


def part_b_autograd() -> None:
    print("\n" + "=" * 70)
    print("B. autograd: requires_grad=True인 텐서로 계산하면, backward()가 '기울기'를 자동으로 구해 준다")
    w = torch.tensor(1.0, requires_grad=True)  # 학습할 파라미터 (하나짜리)
    x, target = 2.0, 6.0  # 입력과 정답. 우리는 w*x == target 이 되는 w(=3)를 찾고 싶다
    pred = w * x
    loss = (pred - target) ** 2  # "얼마나 틀렸나"를 숫자 하나로: 제곱 오차
    print(f"w={w.item()}, pred=w*x={pred.item()}, target={target}, loss=(pred-target)^2={loss.item()}")

    loss.backward()  # loss를 w로 미분한 값(기울기)을 w.grad에 써 준다
    analytic = 2 * (pred.item() - target) * x  # 손으로 미분: d/dw (wx - t)^2 = 2(wx - t) * x
    print(f"w.grad = {w.grad.item()}   (손으로 미분한 값 {analytic} 과 같다)")
    print("기울기의 뜻: w를 아주 조금 '늘리면' loss가 이 비율로 변한다. 음수이므로 w를 늘려야 loss가 준다")

    print("\n  주의: backward()는 grad를 '덮어쓰지' 않고 '더한다'. 그래서 매 step 비워야 한다(zero_grad)")
    loss2 = (w * x - target) ** 2
    loss2.backward()
    print(f"  같은 loss를 한 번 더 backward -> w.grad = {w.grad.item()}  (두 배가 됐다. 실수의 원인)")
    w.grad = None  # optimizer.zero_grad(set_to_none=True)가 하는 일
    print("  w.grad = None 으로 비움")


def part_c_descent(lr: float, steps: int) -> None:
    print("\n" + "=" * 70)
    print(f"C. 기울기 하강: w <- w - lr * grad 를 반복. lr(학습률)={lr}, {steps}번")
    w = torch.tensor(0.0, requires_grad=True)
    x, target = 2.0, 6.0
    for step in range(1, steps + 1):
        loss = (w * x - target) ** 2
        loss.backward()
        with torch.no_grad():  # 파라미터를 고치는 동안은 미분 기록을 끈다 (optimizer.step()의 내부와 같다)
            w -= lr * w.grad
        w.grad = None
        if step <= 5 or step % 5 == 0 or step == steps:
            print(f"  step {step:3d}  w = {w.item():10.4f}   loss = {loss.item():12.4f}")
    print(f"정답 w = {target / x}.  w가 정답에 다가가면 성공, 커지기만 하면 발산(lr이 너무 큼), 거의 안 움직이면 너무 작음")
    print("이 예에서 수학적으로 lr < 0.25 여야 수렴한다(2*x^2 = 8, 1/8 = 0.125의 두 배). LLM에서는 이 경계를 모르니 작게 시작한다(1e-4 등)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.1, help="학습률")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--by-hand", action="store_true", help="md '손으로 계산해 보기'의 정답을 torch로 찍는다")
    ap.add_argument("--by-hand2", action="store_true", help="md '손으로 계산해 보기 2'(층 2개, step 2번)의 정답")
    ap.add_argument("--train-emb", action="store_true", help="--by-hand2에서 E도 갱신한다")
    ap.add_argument("--by-hand3", action="store_true", help="md '손으로 계산해 보기 3'(bias + ReLU)의 정답")
    args = ap.parse_args()
    if args.by_hand:
        part_e_by_hand()
        return
    if args.by_hand2:
        part_f_two_layers(args.train_emb)
        return
    if args.by_hand3:
        part_g_bias_relu()
        return
    part_a_tensor()
    part_b_autograd()
    part_c_descent(args.lr, args.steps)


def part_e_by_hand() -> None:
    """md의 '손으로 계산해 보기'와 같은 숫자. 어휘 3개, 벡터 길이 2, 토큰 2개, 층 하나."""
    print("=" * 70)
    print("E. 손계산 검산: 어휘 3개 · 벡터 길이 2 · 토큰 2개 · 층 하나 (W: [2, 2])")
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)  # 임베딩 테이블 [어휘 3, 2]
    W = torch.tensor([[1.0, 2.0], [0.0, 1.0]], requires_grad=True)  # 층 하나 [출력 2, 입력 2]
    T = torch.tensor([[2.0, 1.0], [1.0, 1.0]])  # 정답 (열 = 토큰)
    ids = [0, 2]

    X = E[ids].T  # [2, L]: 열 l = l번 토큰의 벡터
    X.retain_grad()
    Y = W @ X  # forward
    Y.retain_grad()
    loss = 0.5 * ((Y - T) ** 2).sum()
    print(f"ids = {ids}")
    print(f"X = E[ids]ᵀ =\n{X.detach()}")
    print(f"forward  Y = W·X =\n{Y.detach()}")
    print(f"loss = ½·Σ(Y − T)² = {loss.item()}")

    loss.backward()
    gW, gX = W.grad, X.grad
    assert gW is not None and gX is not None
    print(f"\nd loss/dY = Y − T =\n{Y.grad}")
    print(f"backward ①  d loss/dW = (d loss/dY)·Xᵀ =\n{gW}")
    print(f"backward ②  d loss/dX = Wᵀ·(d loss/dY) =\n{gX}")
    print(f"임베딩 테이블의 기울기 d loss/dE (꺼내 쓴 행만 0이 아니다) =\n{E.grad}")
    gE = E.grad
    assert gE is not None
    with torch.no_grad():
        W_new = W - 0.1 * gW
        E_new = E - 0.1 * gE
        loss_after = 0.5 * ((W_new @ E_new[ids].T - T) ** 2).sum()
    print(f"\nlr 0.1로 한 step: W ← W − 0.1·grad =\n{W_new}")
    print(f"E ← E − 0.1·grad (backward ②의 결과가 여기서 쓰인다) =\n{E_new}")
    print(f"갱신 뒤 다시 forward한 loss = {loss_after.item():.4f}  (갱신 전 {loss.item()})")

    # 수치 검산: W11을 0.01 늘리면 loss가 d loss/dW11 × 0.01 만큼 변해야 한다
    with torch.no_grad():
        W2 = W.clone()
        W2[0, 0] += 0.01
        loss2 = 0.5 * ((W2 @ X - T) ** 2).sum()
    print(f"\n검산: W11을 0.01 늘리면 loss {loss.item()} → {loss2.item():.4f} (변화 {loss2.item() - loss.item():+.4f} ≈ d loss/dW11 × 0.01 = {gW[0, 0].item() * 0.01:+.4f})")


def part_f_two_layers(train_emb: bool) -> None:
    """md '손으로 계산해 보기 2'와 같은 숫자. 성분 3, 토큰 2, 1층 [2,3], 2층 [1,2], lr 0.5, step 2번."""
    print("=" * 70)
    print(f"F. 층 2개 · step 2번 (E는 {'갱신' if train_emb else '얼림'})")
    E = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], requires_grad=train_emb)
    W1 = torch.tensor([[-1.0, -1.0, 0.0], [1.0, 0.0, 1.0]], requires_grad=True)
    W2 = torch.tensor([[1.0, 1.0]], requires_grad=True)
    T = torch.tensor([[3.0, 1.0]])
    ids = [0, 2]
    lr = 0.5
    for step in (1, 2, 3):
        X = E[ids].T
        H = W1 @ X
        H.retain_grad()
        Y = W2 @ H
        loss = 0.5 * ((Y - T) ** 2).sum()
        print(f"\n--- step {step} forward ---")
        print(f"X =\n{X.detach()}\nH = W1·X =\n{H.detach()}\nY = W2·H = {Y.detach().tolist()}   loss = {loss.item()}")
        if step == 3:
            break
        loss.backward()
        gW1, gW2, gH = W1.grad, W2.grad, H.grad
        assert gW1 is not None and gW2 is not None and gH is not None
        print(f"dY = {(Y - T).detach().tolist()}")
        print(f"2층 ① dW2 = dY·Hᵀ  = {gW2.tolist()}")
        print(f"2층 ② dH  = W2ᵀ·dY =\n{gH}")
        print(f"1층 ① dW1 = dH·Xᵀ  =\n{gW1}")
        with torch.no_grad():
            W1 -= lr * gW1
            W2 -= lr * gW2
            if train_emb:
                assert E.grad is not None
                print(f"1층 ② dX → dE =\n{E.grad}")
                E -= lr * E.grad
                E.grad = None
        W1.grad = None
        W2.grad = None
        print(f"갱신: W1 =\n{W1.detach()}\nW2 = {W2.detach().tolist()}")


def part_g_bias_relu() -> None:
    """md '손으로 계산해 보기 3'과 같은 숫자. 예제 2의 설정에 1층 bias와 ReLU를 붙이고 step 1번."""
    print("=" * 70)
    print("G. bias + ReLU: Z = W1·X + b, H = ReLU(Z), Y = W2·H. step 1번, lr 0.5")
    E = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    W1 = torch.tensor([[-1.0, -1.0, 0.0], [1.0, 0.0, 1.0]], requires_grad=True)
    b = torch.tensor([[2.0], [0.0]], requires_grad=True)
    W2 = torch.tensor([[1.0, 1.0]], requires_grad=True)
    T = torch.tensor([[3.0, 1.0]])
    X = E[[0, 2]].T
    lr = 0.5
    for step in (1, 2):
        Z = W1 @ X + b
        Z.retain_grad()
        H = torch.relu(Z)
        H.retain_grad()
        Y = W2 @ H
        loss = 0.5 * ((Y - T) ** 2).sum()
        print(f"\n--- {'step 1' if step == 1 else '갱신 뒤 확인'} forward ---")
        print(f"Z = W1·X + b =\n{Z.detach()}\nM = (Z > 0) =\n{(Z > 0).int()}\nH = ReLU(Z) =\n{H.detach()}")
        print(f"Y = W2·H = {Y.detach().tolist()}   loss = {loss.item()}")
        if step == 2:
            break
        loss.backward()
        gW1, gW2, gb, gH, gZ = W1.grad, W2.grad, b.grad, H.grad, Z.grad
        assert gW1 is not None and gW2 is not None and gb is not None and gH is not None and gZ is not None
        print(f"dY = {(Y - T).detach().tolist()}")
        print(f"dW2 = dY·Hᵀ  = {gW2.tolist()}")
        print(f"dH  = W2ᵀ·dY =\n{gH}")
        print(f"dZ  = dH ⊙ M =\n{gZ}")
        print(f"dW1 = dZ·Xᵀ  =\n{gW1}")
        print(f"db  = dZ 행별 합 = {gb.flatten().tolist()}")
        with torch.no_grad():
            W1 -= lr * gW1
            W2 -= lr * gW2
            b -= lr * gb
        print(f"갱신: W1 =\n{W1.detach()}\nW2 = {W2.detach().tolist()}\nb = {b.detach().flatten().tolist()}")


if __name__ == "__main__":
    main()
