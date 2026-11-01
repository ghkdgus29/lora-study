# lora-study - LoRA SFT를 밑바닥부터 이해하기 위한 단계별 실습

LLM을 LoRA로 파인튜닝하는 실전 학습 스크립트(140줄 남짓)를 **한 줄도 빠짐없이 설명할 수 있게** 되는 것이 목표다.
예제 작업은 "질문과 문서 후보를 주면 문서마다 Y/N으로 답하는 필터"다.
PyTorch 경험이 거의 없고 "loss를 줄이는 방향으로 가중치를 고친다" 정도만 아는 상태에서 시작한다. 전부 맥에서 돈다(작은 모델 Qwen3-0.6B, 1.4GB).

각 단계는 **스크립트(`.py`) + 설명(`.md`)** 한 쌍이다. 순서:
1. md의 "핵심 개념"을 읽는다 (10분)
2. 스크립트를 돌리고 "실행하고 볼 것"을 확인한다
3. "실험해 보기"의 인자를 바꿔 돌리며 경향을 본다 - **여기가 핵심**이다. 표의 결과는 실제로 돌린 값이다
4. "실전 코드에서는"에서 실제 학습 스크립트의 어느 줄에 해당하는지 본다

| 단계 | 주제 | 배우는 코드 조각 | 시간 |
|---|---|---|---|
| [01](steps/01_tensor_autograd.md) | 텐서, autograd, 기울기 하강 | `backward()`, `opt.step()`, `zero_grad()`, lr | 20분 |
| [02](steps/02_training_loop.md) | 학습 루프: AdamW, 배치, accumulation, warmup 스케줄, clipping | `for` 루프 전체, `accum`, `LambdaLR`, `clip_grad_norm_` | 30분 |
| [03](steps/03_cross_entropy.md) | logits, softmax, cross-entropy, 라벨 불균형과 가중치 | `F.cross_entropy`, `weight`, `--y-weight` | 30분 |
| [04](steps/04_next_token.md) | LLM = 다음 토큰 분류기: 토크나이저, logits 모양, greedy, teacher forcing | `AutoTokenizer`, `AutoModelForCausalLM`, `.logits`, `generate` | 30분 |
| [05](steps/05_chat_template.md) | chat template, 프롬프트(P)/답(A) 경계 | `encode()` | 20분 |
| [06](steps/06_answer_loss.md) | 답 토큰에만 loss, `logits_to_keep`, gradient checkpointing | `answer_loss()`, 메모리 설정 3줄 | 30분 |
| [07](steps/07_lora.md) | LoRA 원리(손으로 구현) → PEFT, r/alpha/target_modules, 어댑터 저장 | `LoraConfig`, `get_peft_model`, `save_pretrained` | 30분 |
| [08](steps/08_sft_lab.md) | **실험실**: 미니 필터 SFT를 돌리고 lr/r/step/y_weight/accum을 바꿔 비교 | 전체 | 1~2시간 |
| [09](steps/09_full_pipeline.md) | **전체 조립**: 실전 학습·평가 스크립트 줄 단위 해설, 파라미터 → 경향 요약표 | 전체 | 30분 |

[GLOSSARY.md](GLOSSARY.md)에 용어를 모아 뒀다. 막히는 단어가 나오면 거기서 찾는다.

## 준비

```bash
cd ~/projects/lora-study
uv sync                       # .venv에 torch, transformers, peft, accelerate, matplotlib
uv run steps/01_tensor_autograd.py
```
4단계부터 Qwen3-0.6B를 Hugging Face Hub에서 받는다(`~/.cache/huggingface/hub`, 1.4GB, 이미 받아 뒀다). 맥 GPU(MPS)로 돈다.

## 읽을 때의 태도

- 숫자를 **직접** 바꿔 보고 결과가 md의 "경향"과 맞는지 본다. 안 맞으면 그게 더 중요한 발견이다(잡음인지, md가 틀렸는지).
- "왜 이 값인가"에는 대부분 관례와 메모리 제약이 답이다. 9단계의 요약표가 그 답을 모아 놓은 것이다.
- 8단계 실험실의 결과는 0.6B 모델 + 장난감 데이터라 **방향(경향)만** 믿고 숫자는 믿지 않는다. 실제 35B에서는 스모크 테스트와 val F1로 다시 본다.

## 다 끝나면 할 수 있어야 하는 것

- LoRA SFT 학습 스크립트를 위에서 아래로 읽으며 각 줄의 역할을 말한다.
- 로그의 `loss`, `lr`, `grad_norm`, `tokens_per_s`, `max_mem_gb`를 보고 학습이 정상인지 판단한다.
- 결과가 나쁠 때 lr, step, r, y_weight, 임계값 중 무엇을 먼저 만질지 이유와 함께 고른다.
