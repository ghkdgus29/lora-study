# 4단계: LLM은 "다음 토큰 분류기"다

**목표**: 토크나이저 · `model(...).logits` · greedy 생성 · "위치 t-1이 토큰 t를 예측한다"를 작은 모델(Qwen3-0.6B)로 직접 본다.
이 네 가지를 알면 학습 쪽의 `encode()`와 `answer_loss()`(5, 6단계), 평가 쪽의 점수 매기기(8단계)가 전부 읽힌다.

## 핵심 개념

### 토크나이저
글자를 정수 id로 바꾼다. 모델은 정수만 본다. 한국어는 글자 단위로 잘게 쪼개지는 경우가 많다(`'재', '고', ' 수', '량'`).
- `' Y'`(앞에 공백)와 `'Y'`는 **다른 토큰**이다. 답 `[1] Y`에서 `]` 다음에 오는 것은 `' Y'`(809). 학습 코드가 `y_id = tok.encode(" Y")`로 가중치 줄 토큰을 잡고, 평가 코드가 `" Y"`와 `"Y"`를 모두 Y로 세는 이유.
- 어휘 크기 ≈ 15만(0.6B) / 25만(Qwen3.6). 모델 출력층(lm_head)의 logits 마지막 차원이 이 크기다(모델 쪽은 151,936처럼 조금 더 크게 잡아 둔다 - 여분 자리).

### logits의 모양 `[배치, 길이, 어휘]`
forward 한 번에 **모든 위치**에서 "다음 토큰"의 점수(어휘 전체)가 나온다. 길이 13,000 × 어휘 248,320 × 4바이트 = 13GB. 한 예시의 logits가 이만큼이라 실전 코드는 답 위치 53개만 계산한다(`logits_to_keep`, 6단계).

### greedy 생성
"마지막 위치의 확률 최대 토큰을 붙이고 다시 forward"의 반복. `model.generate(do_sample=False)`가 이것이다. 손으로 쓴 루프와 결과가 같다.
서비스 필터도, 평가 스크립트도 greedy다(temperature 0).

### 학습의 시선: teacher forcing
학습할 때는 토큰을 하나씩 생성하지 않는다. **정답 문장을 통째로 넣고 forward 한 번**으로, 위치 t-1의 logits가 토큰 t를 얼마나 잘 맞혔는지(-log P)를 모든 위치에서 동시에 잰다.
- 그래서 "위치 t-1의 logits ↔ 토큰 t"로 **한 칸 어긋나게** 짝을 짓는다. 마지막 위치의 logits는 짝(정답)이 없어 버린다 → `logits[0, :-1]`과 `ids[0, p_len:]`.
- D의 표에서 `[1] Y\n[2] N\n[3] N`을 보면: 처음에는 `[`, `]` 뒤를 못 맞히다가(P 0.000) 두 번째 줄부터 `[`→`2`→`]`를 0.8~1.0으로 맞힌다. 학습 없이 **문맥만으로 패턴을 따라간 것**(in-context learning). 반면 `]` 뒤에 ` Y`냐 ` N`이냐는 0.02~0.34로 모른다 - 이것이 학습으로 가르칠 부분.

## 실행하고 볼 것

```bash
uv run steps/04_next_token.py
```
- A: `' Y'`와 `'Y'`의 id가 다른지. 한국어가 어떻게 쪼개지는지.
- B: `logits.shape`의 마지막 차원이 어휘 크기인지. 상위 5개 토큰이 그럴듯한지.
- C: 손으로 쓴 루프와 `generate`의 결과가 같은지.
- D: 형식 토큰(`[`, `]`, 숫자, `\n`)은 두 번째 줄부터 거의 맞히고, ` Y`/` N`은 못 맞히는 것.

## 실험해 보기

| 명령 | 볼 것 |
|---|---|
| `--prompt "오늘 점심은" --max-new 12` | 다른 프롬프트의 상위 5개와 생성. greedy는 항상 같은 결과(실행마다 동일) |
| `--prompt "1, 2, 3, 4,"` | 숫자 패턴은 학습 없이도 이어 간다 - D의 `[2]`, `[3]`과 같은 현상 |

**경향**: 모델은 "문맥에 있는 패턴"은 학습 없이 따라 하지만 "문서가 질문에 맞는가"(Y/N)는 문맥만으로 모른다. 파인튜닝은 후자를 가르치는 것이고, 형식(`[n] Y/N` 10줄)은 거의 공짜로 배운다.

## 실전 코드에서는 (학습 · 평가)
```python
tok = AutoTokenizer.from_pretrained(args.model)                                            # A
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
y_id = tok.encode(" Y", add_special_tokens=False)[0]                                      # A: ' Y' = 하나의 토큰
out = model(input_ids=ids, logits_to_keep=..., use_cache=False)                           # B: logits [1, L, 어휘]
logits = out.logits[0, :-1]; targets = ids[0, p_len:]                                     # D: 한 칸 어긋난 짝
model.generate(ids, max_new_tokens=..., do_sample=False, output_logits=True, ...)         # C: greedy + 매 step logits
```
