# 2단계: 학습 루프의 뼈대 - optimizer · 배치 · accumulation · 스케줄 · clipping

**목표**: train_lora.py의 `for` 루프 안에 있는 모든 줄을 파라미터 2개짜리 문제에서 똑같이 돌려 본다. 1단계의 `w -= lr * grad`가 실제로는 어떻게 포장되는지.

## 핵심 개념

### optimizer: SGD와 AdamW
- **SGD**: 1단계 그대로 `w <- w - lr * grad`.
- **Adam(W)**: 파라미터마다 최근 기울기의 평균(방향)과 제곱 평균(크기)을 기억해, **기울기 크기에 상관없이 비슷한 보폭**으로 움직인다.
  그래서 lr의 의미가 "한 step에 파라미터가 대략 lr 정도 움직인다"에 가깝고, 파라미터마다 기울기 크기가 천차만별인 신경망에서 표준이다.
  `W`는 weight decay(가중치를 조금씩 0으로 당기는 규제)를 붙인 것 - train_lora.py는 `weight_decay=0.0`이라 사실상 Adam이다.
- 비용: Adam은 파라미터마다 상태 2개(평균, 제곱 평균)를 fp32로 들고 있다 → 파라미터 1개당 8바이트 추가. **35B 전체를 학습하면 optimizer 상태만 280GB** - LoRA를 쓰는 이유 중 하나(7단계).

### 배치(batch)와 gradient accumulation
- loss를 예시 한 개로 재면 기울기가 그 예시에 휘둘린다(잡음). 여러 개의 평균을 쓰면 안정된다 - 그 묶음이 배치.
- 메모리에 한 번에 못 올리면 **작게 나눠 backward를 여러 번 하고 grad를 누적**한다(1단계: grad는 더해진다). `loss / accum`으로 미리 나누면 정확히 평균이 된다.
- train_lora.py: 예시 길이가 8k~14k 토큰으로 제각각이라 패딩 없이 **1개씩** forward하고 `GRAD_ACCUM=8`마다 step → "배치 8"과 수학적으로 같다.

### 학습률 스케줄: warmup + 선형 감소
- 시작 직후에는 Adam의 "최근 기울기 평균"이 아직 엉성해서 큰 step이 위험하다 → 처음 몇 step은 lr을 0에서 서서히 올린다(**warmup**).
- 끝으로 갈수록 lr을 줄여 미세 조정하고 마지막에 0 근처에서 멈춘다(**선형 감소**).
- `LambdaLR(opt, f)`는 step 번호 `s`를 받아 "기본 lr에 곱할 배수"를 돌려주는 함수 `f`로 스케줄을 만든다. train_lora.py의 람다가 정확히 이것이다:
  `s < warmup`이면 `s / warmup`(0→1), 그 뒤는 `(total - s) / (total - warmup)`(1→0).

### grad clipping
기울기 벡터 전체의 길이(norm)가 `max_norm`을 넘으면 **방향은 두고 길이만** `max_norm`으로 줄인다. 어쩌다 나오는 거대한 기울기 한 방이 학습을 망치는 것을 막는 안전장치. 거의 모든 LLM 학습이 1.0을 쓴다.

## 실행하고 볼 것

```bash
uv run steps/02_training_loop.py
```
- A: `lr` 열이 warmup 동안 올라갔다가 선형으로 내려오는지. w, b가 3, 1로 가는지. `grad_norm`이 어떻게 변하는지.
- B: 32개 한 번과 16개 × 2(각각 /2)의 `w.grad`가 소수점까지 같은지.
- C: clip 뒤 norm이 1.0이 되는지.

## 실험해 보기

기본값은 AdamW, lr 0.2, 60 step, clip 꺼짐(0). 아래는 실제로 돌려 본 결과다(잡음 때문에 숫자는 조금 다를 수 있다).

| 명령 | 관찰되는 경향 |
|---|---|
| `--opt sgd --lr 0.1` | 10 step 만에 w 2.9 도달. 이 문제에서는 SGD가 Adam보다 빠르다(기울기 크기가 그대로 보폭이 되니까) |
| `--opt sgd --lr 0.5` | step 3에 3.5로 지나치고 step 30에 w=-4까지 **크게 진동**. lr이 줄어드는 끝에서야 자리를 잡는다 |
| `--opt sgd --lr 1.0` | **발산** - loss가 10^21까지 커진다(1단계와 같은 이유) |
| `--opt sgd --lr 0.5 --clip 1.0` | 같은 lr인데 진동이 사라지고 step 10에 w 3.0. **clip이 큰 보폭을 막아 준 것** |
| `--opt adamw --lr 0.5` | 진동은 있지만 발산하지 않는다. Adam은 보폭이 lr로 제한된다 |
| `--opt adamw --lr 0.01` | 60 step에 w 0.27 - 한참 모자람. `--steps 2000`으로 늘리면 2.99 도달. **lr과 step 수는 맞교환**이지만 너무 작은 lr은 비효율 |
| `--batch 1` | loss가 71 → 24 → 85로 들쭉날쭉(예시 하나의 잡음). 그래도 60 step에 w 3.2 |
| `--batch 1 --accum 8` | `--batch 8 --accum 1`(기본)과 w, b 궤적이 **소수점까지 같다**(기울기가 같으니까). 단 forward를 8배 하므로 시간은 더 걸린다. 표시되는 loss는 마지막 예시 하나의 값이라 들쭉날쭉하다 |
| `--warmup-ratio 0 --opt adamw --lr 0.5` | 첫 step부터 w가 0.5씩 움직이고 step 10에 3.5로 지나친다. warmup이 있으면(기본 10%) 초반이 완만하다 |
| `--clip 0.1`, `--clip 1.0` | **Adam에서는 거의 차이가 없다** - Adam이 기울기 크기를 어차피 정규화하기 때문. clip은 SGD류에서 크게 작용하고, Adam에서는 "이상한 예시 한 방"을 막는 보험이다 |

**경향 정리**
- lr ↑: 빠르지만 불안정(진동·발산). lr ↓: 안정적이지만 느려서 step이 더 필요.
- accum ↑(= 배치 ↑): 기울기가 안정되고 한 epoch의 step 수가 줄어든다. 같은 데이터로 step이 줄면 "학습량"도 줄어드니 lr을 조금 올리거나 epoch를 늘리기도 한다.
- warmup: 0으로 두면 초반이 거칠다. 3~10%가 관례(train_lora.py는 3%).
- clip 1.0: 거의 고정값. Adam에서는 평소 영향이 없고 이상치 방어용. 로그의 grad_norm이 꾸준히 1을 크게 넘으면 lr이 큰 신호.

## train_lora.py와의 연결
```python
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
for i, idx in enumerate(order[: total_steps * args.accum]):
    loss = answer_loss(...)
    (loss / args.accum).backward()
    if (i + 1) % args.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
```
이 단계의 A와 구조가 글자 단위로 같다. 다른 것은 loss를 어떻게 만드느냐(3, 6단계)와 params가 LoRA 파라미터라는 것(7단계)뿐이다.
