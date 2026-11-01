# 6단계: 답 토큰에만 loss - `answer_loss()` 해부

**목표**: 답 토큰 loss 함수 `answer_loss()` 7줄을 조각내서, 각 줄이 왜 그 모양인지(특히 `logits_to_keep=n_answer + 1`과 `[:-1]`) 숫자로 확인한다. 덤으로 gradient checkpointing이 메모리를 어떻게 줄이는지 본다.

## 핵심 개념

### 한 칸 어긋난 짝 (4단계 D의 복습)
토큰 `P`를 예측하는 것은 위치 `P-1`의 logits다. 답 토큰이 `P .. L-1`(A개)이면 필요한 logits는 위치 `P-1 .. L-2`(A개).
- `logits_to_keep=A+1`: 모델이 **마지막 A+1개 위치**의 logits만 계산한다(위치 `P-1 .. L-1`).
- `[:-1]`: 그중 맨 끝(위치 `L-1`, "`<|im_end|>` 다음"을 예측하는 자리)은 정답이 없으니 버린다 → 위치 `P-1 .. L-2`.
- `targets = ids[0, p_len:]`: 토큰 `P .. L-1`. 길이가 A로 같다.

A에서 전체 logits를 구해 자른 loss(1.5113)와 `logits_to_keep`로 구한 loss가 소수점까지 같다.

### logits_to_keep가 아끼는 메모리
logits는 `[1, L, 어휘]`. L 13,000 × 어휘 248,320 × fp32 4바이트 = **12.9GB**를 예시 하나마다 만들게 된다. 답 위치 53개만 계산하면 53MB. 이것 없이는 96GB GPU에서 35B 모델(72GB)과 긴 예시를 같이 못 돌린다.

### 토큰별 loss를 보면 "무엇을 배우는지"가 보인다 (B)
| 토큰 | 학습 전 P(정답) | 뜻 |
|---|---|---|
| 첫 `[` | 0.000 | 답을 어떻게 시작할지 모른다(`Y`로 시작하려 함). 학습 초반에 금방 배운다 |
| `1`, `]`, 둘째 줄부터의 `[`, 숫자, `]`, `\n` | 0.98~1.00 | 문맥 패턴으로 이미 안다. loss 기여 거의 0 |
| 첫 `\n` | 0.004 | 모델은 `'  \n'`(공백 둘 + 줄바꿈)을 내려 했다 - 맥 확인 때 `MAX_NEW_TOKENS` 64로 10줄이 잘린 원인 |
| ` Y`, ` N` | 0.60 / 0.03 / 0.97 | **판단**. 학습이 실제로 바꾸는 곳. 3번 ` Y`는 1번을 보고 따라 한 것(진짜 판단이 아님) |
| `<|im_end|>` | 1.000 | 끝낼 줄 안다 |

답 52토큰 중 Y/N은 10개뿐이고 나머지는 형식 토큰이라 loss 평균이 빠르게 떨어지는 것은 형식을 배운 것일 수 있다. **loss가 낮아졌다 ≠ Y/N을 잘 고른다** - 그래서 학습이 끝나면 val 데이터로 F1을 따로 잰다(8단계).

### 가중 평균
`(ce * weight).sum() / weight.sum()`: 가중치를 준 토큰이 평균에서 그만큼 여러 번 세어진다. `--y-weight 3`이면 ` Y` 토큰 3개가 9개로 세어지는 셈. 가중치 1이면 보통 평균과 같다(B의 두 값이 같다).

### gradient checkpointing (C)
backward를 하려면 forward 때 각 층의 중간 결과(활성값)를 저장해 둬야 한다. 길이 1,024에서 0.6B도 4.56GB. 길이 13,000 × 35B면 수백 GB.
checkpointing을 켜면 저장하지 않고 backward 때 **다시 계산**한다 → 메모리 0.08GB, 시간은 1.7배(GPU에서는 보통 20~40% 추가). 긴 프롬프트로 큰 모델을 학습할 때는 항상 켠다.
`enable_input_require_grads()`는 입력 임베딩 출력에 `requires_grad`를 켜서 checkpointing 구간을 통과해 기울기가 흐르게 하는 보조 설정이다(LoRA처럼 임베딩은 고정하고 중간 층만 학습할 때 필요).

## 실행하고 볼 것

```bash
uv run steps/06_answer_loss.py
uv run steps/06_answer_loss.py --mem-demo --seq-len 1024      # 1분 안팎
```
- A: 두 loss가 같은지, logits 크기 차이(28MB vs 5MB).
- B: 토큰별 P(정답) - 형식 토큰은 쉽고 Y/N은 어렵다.
- C: checkpointing 켬/끔의 메모리와 시간.

## 실험해 보기

| 명령 | 볼 것 |
|---|---|
| `--y-weight 3` | ` Y` 행의 가중치가 3이 되고 가중 평균이 바뀐다(어느 쪽으로? ` Y`의 loss가 평균보다 낮으면 내려가고 높으면 올라간다) |
| `--mem-demo --seq-len 2048` | 끔의 메모리가 길이에 비례해 2배, 켬은 거의 그대로 |
| `MESSAGES`의 답을 `[1] N\n[2] N\n[3] N`으로 바꾸기 | 1번 ` N`의 P(정답)이 낮아진다(모델은 1번을 Y로 보고 있었다). 2·3번은 1번을 따라간다 |

## 실전 코드에서는
```python
def answer_loss(model, ids, p_len, y_id, y_weight):
    n_answer = ids.shape[1] - p_len
    out = model(input_ids=ids, logits_to_keep=n_answer + 1, use_cache=False)   # A
    logits = out.logits[0, :-1].float()                                        # A: 마지막 하나 버림, fp32로
    targets = ids[0, p_len:]                                                   # A
    ce = F.cross_entropy(logits, targets, reduction="none")                    # B (3단계)
    weight = torch.where(targets == y_id, y_weight, 1.0)                       # B
    return (ce * weight).sum() / weight.sum()                                  # B
```
`.float()`: bf16 logits로 softmax를 하면 정밀도가 모자라 loss가 흔들리므로 fp32로 올려 계산한다(53 × 248k × 4바이트 = 53MB라 부담 없다).
`use_cache=False`: 생성용 KV 캐시를 만들지 않는다(학습에는 필요 없고 메모리만 쓴다).
