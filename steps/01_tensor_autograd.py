"""1단계: 텐서와 자동 미분(autograd), 그리고 "기울기를 따라 내려가기".

모든 학습 루프에 있는 이 세 줄이 무엇을 하는지 가장 작은 예로 본다.
    loss.backward()            # 기울기 계산
    opt.step()                 # 기울기 방향으로 파라미터 수정
    opt.zero_grad()            # 기울기 비우기

실행:  uv run steps/01_tensor_autograd.py
       uv run steps/01_tensor_autograd.py --lr 0.3      # 학습률이 너무 크면?
       uv run steps/01_tensor_autograd.py --lr 0.01     # 너무 작으면?
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
    args = ap.parse_args()
    part_a_tensor()
    part_b_autograd()
    part_c_descent(args.lr, args.steps)


if __name__ == "__main__":
    main()
