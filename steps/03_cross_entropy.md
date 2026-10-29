# 3단계: 분류의 loss - logits · softmax · cross-entropy · 라벨 가중치

**목표**: train_lora.py의 `answer_loss()` 안에 있는 `F.cross_entropy`, `weight`, `(ce * weight).sum() / weight.sum()`이 무엇인지 안다.
LLM의 "다음 토큰 예측"은 어휘(Qwen3.6은 248,320개) 중 하나를 고르는 **분류 문제**이므로, 클래스 3개짜리로 보면 전부 보인다.

## 핵심 개념

### logits → softmax → 확률
- **logits**: 모델이 클래스마다 낸 점수. 아무 실수나 될 수 있다(음수도).
- **softmax**: `exp(점수) / exp(점수)의 합`. 점수를 "합이 1인 확률"로 바꾼다. 점수 차이가 크면 확률이 한쪽으로 쏠린다.
- LLM은 위치마다 어휘 전체에 대한 logits를 낸다 → shape `[배치, 길이, 어휘 수]`. train_lora.py가 `logits_to_keep`를 쓰는 이유가 여기서 나온다(6단계).

### cross-entropy = -log P(정답)
"정답 클래스에 준 확률"만 본다. 확률 1이면 loss 0, 확률 0.04면 3.2, 확률이 0에 가까우면 무한대.
**확신하며 틀리는 것에 큰 벌점**을 주고, 맞는 것에 확신을 더하는 데는 보상이 작다(0.9 → 0.99는 loss 0.105 → 0.01).
`F.cross_entropy(logits, target)`는 softmax와 -log를 한 번에, 수치적으로 안전하게 계산한다.

train_lora.py 로그의 `loss`가 바로 이것이다: 답 토큰 52개마다 -log P(정답 토큰)의 (가중) 평균. loss 0.1이면 토큰마다 평균 확률 약 0.9.

### 기울기 = softmax - one-hot
cross-entropy를 logits로 미분하면 정확히 `확률 - 정답 one-hot`이다. 정답 클래스만 음수(점수를 올려라), 나머지는 양수(내려라).
학습이 하는 일은 결국 "정답 토큰의 점수를 올리고 나머지를 내린다"의 반복이다.

### 라벨 불균형과 가중치
필터 데이터는 라벨의 88%가 N이다. 가중치 없이 학습하면 모델은 "N이라고 하면 대체로 맞는다"는 쪽으로 기운다.
`weight = where(target == Y, y_weight, 1)` 로 Y 예시의 loss를 키우면, Y를 놓치는 것이 더 아프므로 P(Y)가 전체적으로 위로 밀린다.

## 실행하고 볼 것

```bash
uv run steps/03_cross_entropy.py
```
- A: 손으로 계산한 softmax, -log P가 `F.cross_entropy`와 같은지. 정답이 0번일 때(0.24)와 2번일 때(3.24)의 loss 차이.
- B: `logits.grad`가 `softmax - onehot`과 같은지.
- C: 가중치 1에서 Y 예시의 평균 P(Y)가 0.55, N 예시에서 0.05. 임계값에 따라 precision/recall이 어떻게 맞교환되는지.

## 실험해 보기 - Y 가중치 (실제 돌린 결과)

| 가중치 | Y 예시의 평균 P(Y) | N 예시의 평균 P(Y) | 임계값 0.5에서 P / R / F1 | F1이 가장 좋은 임계값 |
|---|---|---|---|---|
| `--y-weight 1` | 0.55 | 0.05 | 0.75 / 0.57 / 0.65 | 0.3 (F1 0.69) |
| `--y-weight 3` | 0.70 | 0.10 | 0.61 / 0.75 / 0.67 | 0.7 (F1 0.68) |
| `--y-weight 9` | 0.81 | 0.19 | 0.44 / 0.88 / 0.58 | 0.7 (F1 0.68) |

**경향**
- 가중치 ↑ → P(Y) 전체가 위로 밀린다(Y 예시도, N 예시도). 같은 임계값에서 recall ↑, precision ↓.
- **임계값을 훑어서 고르면 최고 F1은 가중치와 거의 무관하다**(0.68~0.69). 가중치는 "P(Y)의 눈금"을 옮기는 것이고 임계값이 그 눈금을 다시 자르기 때문.
- 그러면 가중치는 언제 중요한가: **임계값을 못 고르는 상황** - 서비스는 모델이 greedy로 낸 `Y`/`N` 글자를 그대로 쓴다(= 임계값 0.5 고정). 이때 가중치 1이면 recall 0.57, 가중치 3이면 0.75.
- 너무 큰 가중치(9)는 임계값 0.5에서 precision이 0.44로 떨어진다. 필터는 precision이 떨어지면 답변에 엉뚱한 문서가 섞이므로 과하면 해롭다.

이것이 README(filter_ft)에 적은 "첫 학습의 val Y 확률 분포를 보고 가중치를 정한다"의 뜻이다: 가중치 1로 학습한 모델의 P(Y) 분포를 보고, greedy(0.5)에서 Y를 너무 안 내면 가중치를 올리거나 임계값을 내린다.

## train_lora.py와의 연결
```python
ce = F.cross_entropy(logits, targets, reduction="none")   # 답 토큰마다 -log P(정답)   <- A
weight = torch.where(targets == y_id, y_weight, 1.0)       # ' Y' 토큰만 가중치         <- C
return (ce * weight).sum() / weight.sum()                  # 가중 평균
```
`targets`가 토큰 id(어휘 248,320개 중 하나)이고 클래스가 248,320개라는 것만 다르다. 주의: 답 토큰 52개 중 ' Y'/' N'은 10개뿐이고 나머지 42개는 `[`, `1`, `]`, `\n` 같은 형식 토큰이다 - 이것들은 금방 loss 0이 되고, 실제 학습은 Y/N 10개에서 일어난다(6단계에서 토큰별 loss로 확인).
