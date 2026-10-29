# 1단계: 텐서와 자동 미분 - "학습"이란 기울기를 따라 내려가는 것

**목표**: `loss.backward()`, `opt.step()`, `opt.zero_grad()` 세 줄이 하는 일을 숫자 하나짜리 예로 몸으로 익힌다.

## 핵심 개념

### 텐서(tensor)
숫자 배열이다. 숫자 하나(스칼라)도, 표(2차원)도, 더 높은 차원도 전부 텐서다. PyTorch에서 **모델의 파라미터도, 입력 토큰 id도, loss도 전부 텐서**다.
`shape`(모양)과 `dtype`(숫자 형식)을 항상 의식한다.
- train_lora.py의 `ids = torch.tensor(ids, device=device)[None]` → 토큰 id 리스트를 텐서로 만들고 `[None]`으로 맨 앞에 "배치 1개" 차원을 붙인다(shape `[1, L]`).
- `dtype=torch.bfloat16` → 숫자 하나를 2바이트로 저장(fp32의 절반). 정밀도는 낮지만 35B 모델을 72GB에 넣으려면 필요하다.

### 파라미터와 loss
- **파라미터**: 학습으로 고쳐 나갈 숫자(예제에서는 `w` 하나, 35B 모델에서는 350억 개).
- **loss(손실)**: "지금 얼마나 틀렸나"를 숫자 **하나**로 만든 것. 이 숫자를 줄이는 게 학습의 전부다.

### 기울기(gradient)와 autograd
기울기 = "파라미터를 아주 조금 늘리면 loss가 얼마나 변하나"(미분값). 음수면 파라미터를 늘려야 loss가 줄고, 양수면 줄여야 한다.
`requires_grad=True`인 텐서로 계산한 뒤 `loss.backward()`를 부르면 PyTorch가 계산 과정을 거꾸로 따라가(역전파) 모든 파라미터의 기울기를 `.grad`에 채워 준다.
우리가 미분 공식을 쓸 필요가 없다 - 이것이 autograd.

### 기울기 하강(gradient descent)
```
w <- w - lr * grad
```
기울기의 **반대 방향**으로 `lr`(학습률)만큼 간다. 이것이 `optimizer.step()`의 가장 단순한 형태(SGD)다.

### zero_grad
`backward()`는 `.grad`를 덮어쓰지 않고 **더한다**. 그래서 한 step 끝나면 비워야 한다. train_lora.py는 이 "더해지는" 성질을 일부러 이용한다(2단계의 gradient accumulation).

## 실행하고 볼 것

```bash
uv run steps/01_tensor_autograd.py
```
- B에서 `w.grad`가 손으로 미분한 값과 같은지.
- B에서 backward를 두 번 하면 grad가 두 배가 되는지(그래서 zero_grad가 필요).
- C에서 w가 정답 3에 다가가는지.

## 실험해 보기 - 학습률(lr)의 영향

| 명령 | 기대하는 경향 |
|---|---|
| `--lr 0.01` | 수렴하지만 느리다(20 step에 아직 멀었다) |
| `--lr 0.1` (기본) | 몇 step 만에 3에 도달 |
| `--lr 0.2` | 왔다 갔다 하면서(진동) 수렴 |
| `--lr 0.3` | loss가 커지기만 한다 = **발산** |

**경향**: 학습률은 "한 걸음 크기". 너무 작으면 느리고, 너무 크면 정답을 지나쳐 튕겨 나간다. LLM에서는 수렴 경계를 미리 알 수 없어서 작게(1e-4, 1e-5) 시작하고,
loss가 오르거나 `nan`이 되면 줄인다. train_lora.py의 `--lr` 기본값 1e-4가 LoRA에서 흔히 쓰는 값이다.

## train_lora.py와의 연결
```python
loss = answer_loss(...)        # loss 계산 (6단계에서)
(loss / args.accum).backward() # 기울기 계산 <- 이 단계의 B
opt.step()                     # w <- w - lr * grad (의 발전형, 2단계) <- 이 단계의 C
opt.zero_grad(set_to_none=True)# grad 비우기 <- 이 단계의 B 끝
```
