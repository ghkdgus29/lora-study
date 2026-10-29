# 9단계: train_lora.py · eval_filter.py 줄 단위 해설

**목표**: 1~8단계를 끝낸 뒤 실제 스크립트(`rag-portfolio/scripts/rag_optimize/filter_ft/train_lora.py`, `eval_filter.py`)를 처음부터 끝까지 읽고 "이 줄은 왜 있나"를 전부 말할 수 있게 한다.
각 블록 옆에 어느 단계에서 배웠는지 적었다. 8단계의 `08_sft_lab.py`는 이 스크립트를 작은 데이터용으로 옮긴 것이라 같은 이름의 함수가 그대로 있다.

## train_lora.py

### 1. `encode(tok, messages)` → 5단계
```python
kw = dict(tokenize=True, enable_thinking=False, return_dict=False)
full = tok.apply_chat_template(messages, **kw)                                   # system+user+assistant 전체 토큰
prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True, **kw)  # 답 직전까지
assert full[: len(prompt)] == prompt                                             # 앞부분이 같아야 P/A로 자를 수 있다
end = max(i for i, t in enumerate(full) if t == tok.convert_tokens_to_ids("<|im_end|>"))
return full[: end + 1], len(prompt)                                              # (<|im_end|>까지, P 길이)
```
- `enable_thinking=False`: 서비스가 thinking을 끄고 부르므로 학습 데이터도 같은 모양(빈 `<think>` 블록 포함)으로.
- 마지막 `\n`은 버린다(모델은 `<|im_end|>`에서 멈춘다).

### 2. `answer_loss(model, ids, p_len, y_id, y_weight)` → 3, 4, 6단계
```python
n_answer = ids.shape[1] - p_len                                            # 답 토큰 수 A (약 52)
out = model(input_ids=ids, logits_to_keep=n_answer + 1, use_cache=False)   # 마지막 A+1 위치의 logits만 (13GB -> 53MB)
logits = out.logits[0, :-1].float()                                        # 위치 P-1..L-2 (맨 끝은 정답이 없어 버림), fp32
targets = ids[0, p_len:]                                                   # 토큰 P..L-1
ce = F.cross_entropy(logits, targets, reduction="none")                    # 토큰마다 -log P(정답)
weight = torch.where(targets == y_id, y_weight, 1.0)                       # ' Y' 토큰만 가중치
return (ce * weight).sum() / weight.sum()                                  # 가중 평균
```

### 3. 인자 → 2, 7, 8단계
| 인자 | 기본 | 뜻 | 바꾸면 |
|---|---|---|---|
| `--model` | Qwen3.6-35B-A3B | 기반 모델 | 맥 확인은 Qwen3-0.6B |
| `--lora-experts` | 꺼짐 | 256개 expert 파라미터에도 LoRA(`target_parameters`) | 학습 파라미터·메모리·시간 ↑. 스모크 테스트에서 비교 |
| `--y-weight` | 1 | ' Y' 토큰 loss 가중치 | ↑ → P(Y) 전체가 위로(recall ↑ precision ↓). 첫 학습의 val 분포 보고 정한다 |
| `--lr` | 1e-4 | 학습률 | 2단계. LoRA는 전체 학습보다 큰 값(1e-4~2e-4)이 보통 |
| `--accum` | 8 | optimizer step당 예시 수 | ↑ 안정, step 수 ↓ |
| `--max-steps` | 없음 | step 제한 | 스모크 테스트용 |
| `--limit` | 없음 | 앞 N개 예시만 | 맥 확인용 |
| `--save-every` | 50 | 어댑터 저장 주기 | 중간에 끊겨도 이어 쓸 수 있게 |
| `--seed` | 0 | 섞는 순서 고정 | 재현용 |

### 4. 토크나이저와 데이터 → 4, 5단계
```python
tok = AutoTokenizer.from_pretrained(args.model)
y_id = tok.encode(" Y", add_special_tokens=False); assert len(y_id) == 1; y_id = y_id[0]   # ' Y'가 토큰 하나인지
rows = load_jsonl(args.data / "train.jsonl")[: args.limit]
examples = [encode(tok, r["messages"]) for r in rows]                                     # 전부 미리 토큰화 (2,950개, 24.7M 토큰)
```

### 5. 모델 읽기와 메모리 설정 → 4, 6단계
```python
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
```
- `AutoModelForCausalLM`: Qwen3.6-35B-A3B는 비전 모델까지 포함한 저장소인데, 이 클래스로 읽으면 텍스트 부분(`Qwen3_5MoeForCausalLM`)만 올라간다.
- `dtype=torch.bfloat16`: 2바이트/파라미터 → 35B ≈ 72GB. fp32면 140GB라 못 올린다.
- `device_map=device`: 전부 한 GPU에.
```python
model.config.use_cache = False                                                            # 생성용 캐시 끔
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # 활성값 저장 안 함(6단계 C)
model.enable_input_require_grads()                                                        # checkpointing 구간으로 기울기가 흐르게
```

### 6. LoRA → 7단계
```python
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=[q/k/v/o_proj, in_proj_qkv, in_proj_z, out_proj, gate/up/down_proj],
                  target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"] if args.lora_experts else None)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]    # LoRA A, B만
```
- Qwen3.6의 층: 40층 중 30층이 Gated DeltaNet(`in_proj_qkv, in_proj_z, out_proj`), 10층이 attention(`q/k/v/o_proj`). 모든 층에 MoE MLP(공유 expert `gate/up/down_proj` + expert 256개).
- expert 256개는 `(256, …)` 3차원 파라미터 하나라 `target_modules`로는 못 잡고 `target_parameters`가 필요하다(dropout 0 필수).

### 7. optimizer와 스케줄 → 2단계
```python
total_steps = min(len(examples) // args.accum, args.max_steps or 10**9)    # 2,950 // 8 = 368
warmup = max(1, int(total_steps * 0.03))                                    # 11
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = LambdaLR(opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
```

### 8. 학습 루프 → 1, 2단계
```python
order = list(range(len(examples))); random.Random(args.seed).shuffle(order)   # 순서 섞기(같은 섹션 질문이 몰리지 않게)
model.train()
for i, idx in enumerate(order[: total_steps * args.accum]):                 # 1 epoch, 나머지는 버림
    ids, p_len = examples[idx]
    ids = torch.tensor(ids, device=device)[None]                             # [1, L]
    loss = answer_loss(model, ids, p_len, y_id, args.y_weight)
    (loss / args.accum).backward()                                           # 기울기 누적
    if (i + 1) % args.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        (로그: step, loss(최근 accum개 평균), lr, tokens_per_s, elapsed_min, max_mem_gb)
        if opt_step % args.save_every == 0: model.save_pretrained(args.out)
model.save_pretrained(args.out)
```
- `model.train()`: dropout 같은 "학습 모드" 동작을 켠다(여기선 dropout 0이라 효과 없지만 관례).
- `tokens_per_s`: 본 학습 비용 어림용(24.7M 토큰 ÷ 이 값).
- `max_mem_gb`: `torch.cuda.max_memory_allocated()`. 96GB에 들어가는지.

### 9. 로그 읽는 법 → 2, 6, 8단계
- `loss`: 답 토큰 52개의 가중 cross-entropy 평균. 형식 토큰 42개는 금방 0이 되므로 **loss가 0.1 아래로 내려가도 Y/N을 잘 고른다는 뜻은 아니다**. sid F1은 eval_filter.py로.
- loss가 오르거나 `nan`: lr을 낮춘다. 초반 수십 step은 warmup이라 원래 천천히 내려간다.
- `lr`: 11 step까지 올라가고 그 뒤 선형 감소. 끝에서 0.

## eval_filter.py

### `score` → 4, 5단계
```python
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
if args.adapter: model = PeftModel.from_pretrained(model, str(args.adapter))                 # 어댑터 붙이기(없으면 학습 전 모델)
prompt = tok.apply_chat_template(r["messages"][:2], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
out = model.generate(ids, max_new_tokens=128, do_sample=False, output_logits=True, return_dict_in_generate=True, pad_token_id=...)
```
- greedy로 `[1] Y\n[2] N…`을 만들고, `output_logits=True`로 매 step의 logits를 받는다.
- `doc_scores()`: 생성 토큰 중 Y/N 토큰 자리마다 softmax에서 P(Y)/(P(Y)+P(N))을 읽는다. 10줄·번호 1~10이 아니면 `None`(형식 실패).
- 앞 줄의 Y/N은 모델이 낸 대로 둔다(서비스 조건과 같다). 그래서 점수는 "앞 줄들이 그렇게 쓰였을 때"의 조건부 확률.

### `report` → 3단계
- `pick()`: 점수 ≥ 임계값인 문서. 없으면 최대 하나, 점수 없으면 1번(서비스 fallback).
- `row_metrics()`: 후보 안 정답 번호 vs 고른 번호의 F1·precision·recall·exact (dspy_filter/metric.py와 같다).
- 임계값을 안 주면 0.05~0.95를 훑어 F1 최대를 고른다(val용). 3단계 C에서 본 "가중치와 임계값은 같은 손잡이"가 여기서 쓰인다.

## 파라미터 → 경향 요약표

| 파라미터 | ↑ 하면 | ↓ 하면 | train_lora.py 값과 이유 |
|---|---|---|---|
| lr | 빨리 배우지만 진동·발산, 과적합 빠름 | 느림, step 더 필요 | 1e-4: LoRA 관례. loss가 오르면 5e-5 |
| step 수 / epoch | 더 배움, 과적합 위험 | 덜 배움 | 1 epoch(368 step). val F1을 보고 2 epoch 고려 |
| accum (배치) | 기울기 안정, step 수 ↓ | 잡음 ↑, step ↑ | 8: 예시 하나가 길어 1개씩 forward |
| warmup | 초반 안정 | 초반 거침 | 3% |
| clip | (Adam에선 거의 무관) | 이상치 방어 약화 | 1.0 고정 |
| r | 표현력·메모리·어댑터 ↑ | 적은 데이터에 유리, 빠름 | 16: Y/N 판단은 단순한 작업 |
| alpha | 보정 세기 ↑ (lr ↑ 비슷) | 보정 약함 | 32 = 2r |
| target_modules | 더 많은 층 조정 | 일부만 | 전부(attention + DeltaNet + 공유 expert) |
| y_weight | recall ↑ precision ↓ | 반대 | 1. 첫 결과의 P(Y) 분포 보고 결정 |
| dropout | 과적합 ↓, 학습 느림 | - | 0 (expert target_parameters 조건) |
| gradient checkpointing | 메모리 ↓ 시간 ↑ | - | 켬 (13k 토큰 × 35B) |
| logits_to_keep | - | - | A+1: logits 13GB → 53MB |
