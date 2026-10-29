# 5단계: chat template - 메시지가 토큰이 되고, 프롬프트(P)와 답(A)의 경계가 생기는 곳

**목표**: train_lora.py의 `encode()` 6줄을 한 토큰씩 눈으로 확인한다. 3-1단계 데이터(`messages`)가 모델에 어떤 모양으로 들어가는지, "답 토큰에만 loss"가 무엇을 뜻하는지.

## 핵심 개념

### chat template (ChatML)
LLM은 "system/user/assistant"를 모른다. 토크나이저에 들어 있는 템플릿이 역할마다 `<|im_start|>역할\n내용<|im_end|>\n`으로 **하나의 긴 텍스트**로 펼친다. `<|im_start|>`, `<|im_end|>`는 각각 하나의 특수 토큰이다.
- `<|im_end|>`가 곧 eos(생성 종료) 토큰이다. 모델이 이것을 내면 `generate`가 멈춘다.
- **학습과 서비스의 프롬프트 모양이 같아야 한다.** 서비스 필터는 thinking을 끄고 부르므로, 학습 데이터도 `enable_thinking=False`로 만든다 → assistant 턴 앞에 빈 `<think>\n\n</think>\n\n`가 들어간 모양.

### add_generation_prompt
system+user까지만 넣고 `add_generation_prompt=True`를 주면 `<|im_start|>assistant\n<think>\n\n</think>\n\n`까지 붙여 준다 = "여기서부터 답을 써라". 서비스에서 모델을 부를 때 넣는 입력이 정확히 이것이고, eval_filter.py `score()`도 이것을 넣는다.

### prefix 성질과 P/A 경계
전체 시퀀스(`full`)의 앞 `len(prompt)`개 토큰이 `prompt`와 같다. 그래서 "앞 P개 = 프롬프트(읽기만), 그 뒤 = 답(배우기)"으로 자를 수 있다. 템플릿에 따라 이 성질이 깨질 수 있어 train_lora.py는 `assert`로 확인한다(0.6B와 35B 모두 통과).

### 마지막 `\n` 버리기
템플릿은 `<|im_end|>` 뒤에 `\n`을 하나 더 붙인다. 모델은 `<|im_end|>`에서 멈추므로 그 뒤를 배울 필요가 없다 → `full[:end+1]`.

### 왜 프롬프트에는 loss를 안 거나
실제 데이터는 프롬프트 8,300토큰 vs 답 52토큰. 전부에 loss를 걸면 학습의 99%가 "매뉴얼 본문 외우기"가 된다. 답 토큰에만 loss를 걸면 모델은 프롬프트를 읽고 **답 쓰는 법**만 배운다(SFT의 표준). 메모리 면에서도 답 위치의 logits만 계산하면 되므로 `logits_to_keep`이 가능해진다(6단계).

## 실행하고 볼 것

```bash
uv run steps/05_chat_template.py
```
- A: 펼쳐진 텍스트에서 `<|im_start|>`/`<|im_end|>`, 빈 `<think>` 블록의 위치.
- B: `full[:len(prompt)] == prompt`가 `True`.
- C: 69번 토큰 `[`부터 답(A), 78번 `<|im_end|>`가 답의 끝, 79번 `\n`은 버림. 답 10토큰 중 Y/N은 2개뿐.
- D: 프롬프트/답 토큰 수 비율.

## 실험해 보기

| 해 볼 것 | 볼 것 |
|---|---|
| `MESSAGES`의 assistant 내용을 `"[1] Y\n[2] N\n[3] N"`으로 바꿔 실행 | A가 15토큰(줄마다 5토큰: `[`, 숫자, `]`, ` Y/N`, `\n`). 10줄이면 약 52토큰 - README의 "답 52토큰"의 출처 |
| user 내용을 길게 바꿔 실행 | P만 늘고 A는 그대로. loss가 걸리는 토큰 수는 문서 길이와 무관하다 |
| `--model Qwen/Qwen3.6-35B-A3B` | 토크나이저만 받는다(가중치 아님, 이미 캐시에 있음). 특수 토큰 id가 다르고(`<|im_end|>`=248046) 템플릿 모양은 같다 |

## train_lora.py와의 연결
```python
def encode(tok, messages):
    kw = dict(tokenize=True, enable_thinking=False, return_dict=False)
    full = tok.apply_chat_template(messages, **kw)                                  # A, B
    prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True, **kw) # B
    assert full[: len(prompt)] == prompt                                            # B
    end = max(i for i, t in enumerate(full) if t == tok.convert_tokens_to_ids("<|im_end|>"))  # C
    return full[: end + 1], len(prompt)                                             # C
```
`return_dict=False`: transformers 5에서 `tokenize=True`가 기본으로 dict(`BatchEncoding`)를 돌려주므로 토큰 리스트를 받으려면 필요하다(3-1단계에서 토큰 통계가 전부 "2"로 나왔던 버그의 원인).
