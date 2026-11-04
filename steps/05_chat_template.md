# 5단계: chat template - 메시지가 토큰이 되고, 프롬프트(P)와 답(A)의 경계가 생기는 곳

**목표**: 학습 데이터를 토큰으로 바꾸는 `encode()` 6줄을 한 토큰씩 눈으로 확인한다. 학습 데이터(`messages`)가 모델에 어떤 모양으로 들어가는지, "답 토큰에만 loss"가 무엇을 뜻하는지.

## 핵심 개념

### chat template (ChatML)
LLM은 "system/user/assistant"를 모른다. 토크나이저에 들어 있는 템플릿이 역할마다 `<|im_start|>역할\n내용<|im_end|>\n`으로 **하나의 긴 텍스트**로 펼친다. `<|im_start|>`, `<|im_end|>`는 각각 하나의 특수 토큰이다.
- `<|im_end|>`가 곧 eos(생성 종료) 토큰이다. 모델이 이것을 내면 `generate`가 멈춘다.
- **학습과 서비스의 프롬프트 모양이 같아야 한다.** 서비스 필터는 thinking을 끄고 부르므로, 학습 데이터도 `enable_thinking=False`로 만든다 → assistant 턴 앞에 빈 `<think>\n\n</think>\n\n`가 들어간 모양.

### add_generation_prompt
system+user까지만 넣고 `add_generation_prompt=True`를 주면 `<|im_start|>assistant\n<think>\n\n</think>\n\n`까지 붙여 준다 = "여기서부터 답을 써라". 서비스에서 모델을 부를 때 넣는 입력이 정확히 이것이고, 평가할 때도 이것을 넣는다.

### prefix 성질과 P/A 경계
학습 데이터 한 건을 토큰으로 바꾸면 질문과 답이 한 줄로 이어진다. loss는 답 토큰에만 걸어야 하므로, 이 줄의 어디서 답이 시작되는지 알아야 한다. 이 경계를 구하려고 `encode()`는 토큰화를 두 번 한다.

- `full`: system + user + assistant(답)까지 전부 토큰화 (`messages` 전체)
- `prompt`: system + user + "여기서부터 답을 써라"까지 토큰화 (`messages[:2]`, `add_generation_prompt=True`)

아래는 토큰을 단순화해 그린 그림이다.

```text
prompt: [시작][system][질문][assistant 시작]                       ← 4개
full:   [시작][system][질문][assistant 시작][답1][답2][답3][im_end]
        └──────── 앞 4개가 prompt와 동일 ────┘└──── 답 ────┘
```

`full`의 앞 `len(prompt)`개 토큰이 `prompt`와 똑같다. 앞부분이 일치한다는 뜻에서 이것을 **prefix(접두사) 성질**이라 부른다. 이 성질이 성립하면 경계는 숫자 하나로 정해진다.

- `P = len(prompt)` (위 그림에서는 4)
- `full[:P]` = 프롬프트. 모델이 읽기만 한다(loss 없음).
- `full[P:]` = 답. 모델이 배운다(loss 있음).

`encode()`가 토큰 리스트와 함께 `len(prompt)`를 돌려주는 이유가 이것이고, 실행 결과 C의 "69번 토큰 `[`부터 답"이 바로 이 경계다.

**P는 찾아내는 값이 아니라 재는 값이다.** `encode()`는 `full`을 훑어서 경계를 탐지하지 않는다. `messages[:2]`(답을 뺀 부분)를 템플릿에 따로 넣어 토큰으로 만들고, 그 길이를 P로 쓴다. 반면 답의 끝(`end`)은 `full`을 실제로 훑어 마지막 `<|im_end|>`의 위치를 찾는다. 시작은 "프롬프트만 따로 렌더링해서 길이를 센다", 끝은 "검색한다"로 방법이 다르다.

**왜 `assert`로 확인하나**: 앞 문단은 "`full`의 앞 P개 = `prompt`니까 P에서 자르면 된다"였다. 그런데 이것이 항상 참은 아니다.

1. **보장이 없는 이유.** `full`은 messages 3개(system, user, assistant)를 넣어 렌더링한 것이고, `prompt`는 2개(system, user)에 `add_generation_prompt=True`를 줘서 렌더링한 것이다. 둘의 앞부분이 같으려면 템플릿이 두 상황에서 같은 글자를 써야 한다. 즉 "assistant 메시지 앞에 붙이는 머리말"과 "`add_generation_prompt=True`일 때 끝에 붙이는 머리말"이 글자 단위로 똑같아야 한다. 이건 템플릿 작성자가 그렇게 짜 놓았느냐의 문제이지 자동으로 보장되는 규칙이 아니다.

2. **어긋나는 예시.** `prompt`를 만들 때는 끝에 `<|im_start|>assistant\n<think>\n\n</think>\n\n`을 붙였는데, `full`을 만들 때는 assistant 메시지 앞에 `<|im_start|>assistant\n`까지만 붙이고 바로 답을 쓰는 템플릿이라면 이렇게 된다.

   ```text
   prompt: [시작][system][질문][assistant 시작][<think>][</think>]
   full:   [시작][system][질문][assistant 시작][답1][답2]...
                                               ↑ 여기서 어긋남
   ```

   `prompt`에는 `<think>`, `</think>` 토큰이 있어 예를 들어 69개인데, `full`에서는 그 자리에 답의 첫 토큰이 와 있다. 앞 67개는 같고 68번부터 다르다.

3. **이 상태로 자르면.** P=69로 자르면 `full[:69]`에 답의 첫 두 토큰(`[`, `1`)이 섞여 들어가 loss에서 빠진다. 모델은 답의 시작을 배우지 못한다. 반대로 `prompt`가 더 짧은 경우면 프롬프트 끝 토큰이 `full[P:]`에 들어가서 모델이 프롬프트 글자를 "답"으로 배운다.

4. **에러 없이 끝난다는 점이 더 큰 문제다.** 경계가 틀어져도 코드는 숫자 하나(P)로 자를 뿐이라 예외가 나지 않는다. 학습은 끝까지 돌고 loss도 내려가고 결과 모델만 이상하다. 그때 가서 데이터·하이퍼파라미터·모델 중 무엇이 문제인지 찾기 시작하면 원인이 이 한 줄이라는 걸 알기 어렵다.

5. **그래서 `assert`.** `assert full[: len(prompt)] == prompt`는 "잘라도 되는 조건"을 매 샘플마다 검사하고 틀리면 그 자리에서 멈춘다. 비용은 리스트 비교 한 번이고, 위의 조용한 오염을 막는다. 0.6B와 35B 토크나이저로 실제 돌려 봤더니 둘 다 통과했으므로 이 프로젝트의 두 모델에서는 P로 잘라도 안전하다.

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

## 실전 코드에서는
```python
def encode(tok, messages):
    kw = dict(tokenize=True, enable_thinking=False, return_dict=False)
    full = tok.apply_chat_template(messages, **kw)                                  # A, B
    prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True, **kw) # B
    assert full[: len(prompt)] == prompt                                            # B
    end = max(i for i, t in enumerate(full) if t == tok.convert_tokens_to_ids("<|im_end|>"))  # C
    return full[: end + 1], len(prompt)                                             # C
```
`return_dict=False`: transformers 5에서 `tokenize=True`가 기본으로 dict(`BatchEncoding`)를 돌려주므로 토큰 리스트를 받으려면 필요하다. 빼먹으면 `len()`이 dict의 키 개수(2)를 돌려줘서 토큰 통계가 전부 "2"로 나오는 버그가 생긴다.
