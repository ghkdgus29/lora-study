# 7단계: LoRA - 큰 가중치는 얼리고 작은 보정 행렬만 학습한다

**목표**: `LoraConfig`, `get_peft_model`, `print_trainable_parameters`, `save_pretrained`가 무엇을 하는지, 그리고 `r`, `alpha`, `target_modules`가 무엇을 바꾸는지 안다.

## 핵심 개념

### 왜 전체를 학습하지 않나
파라미터 하나를 Adam으로 학습하려면 가중치 외에 **기울기 + Adam 상태 2개 = 파라미터당 12바이트(fp32)**가 더 든다. 35B면 420GB. 96GB GPU 한 장에는 가중치(72GB)만 겨우 들어간다.
또 데이터가 2,950문항뿐이라 350억 개를 전부 움직이면 금방 과적합한다.

### LoRA(Low-Rank Adaptation)
선형층 `y = W x`에 보정을 더한다: `y = W x + B (A x) * (alpha / r)`.
- `W` (예: 2048×1024)는 **얼린다**(requires_grad=False). 학습하지 않고 메모리에 읽기만.
- `A` (r×1024), `B` (2048×r)만 학습. `r`이 16이면 파라미터 수가 W의 1~2%.
- **`B`는 0으로 시작** → 학습 전에는 보정이 0이라 원래 모델과 출력이 완전히 같다(A의 `allclose True`). 거기서 조금씩 벗어난다.
- 학습 결과물 = A, B들만 모은 **어댑터**(40MB; 35B에서는 수백 MB). 서비스는 원래 가중치 + 어댑터를 같이 읽는다.

### r (rank)
보정 행렬의 "폭". 파라미터 수가 r에 비례한다(0.6B: r 4 → 2.5M, 16 → 10M, 64 → 40M).
- 작으면: 바꿀 수 있는 것이 적다 → 과적합 덜하고 빠르다. 간단한 작업(형식·분류)이면 8~16으로 충분한 경우가 많다.
- 크면: 표현력 ↑, 메모리·어댑터 크기 ↑. 데이터가 적으면 이득이 거의 없다.
- train_lora.py의 16은 "Y/N 판단"이 어려운 작업은 아니라고 보고 고른 보수적인 값.

### alpha와 scale = alpha / r
보정에 곱하는 배율. `alpha = 2r`(scale 2)이 관례이고 train_lora.py도 그렇다(16/32). r을 바꾸면 alpha도 같은 비율로 바꿔야 "보정의 세기"가 유지된다.
scale을 키우는 것은 lr을 키우는 것과 비슷한 효과(보정이 출력에 더 크게 반영).

### target_modules
LoRA를 붙일 nn.Linear의 **이름 끝부분**. B에서 본 층 하나의 목록(`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)에 전부 붙이면 10M, attention의 q/v에만 붙이면 2.3M.
- 원 논문은 q, v에만 붙였지만 요즘은 전부 붙이는 쪽이 보통 더 좋다(같은 파라미터 예산이면 r을 줄이고 모듈을 늘리는 편이 낫다는 보고가 많다).
- Qwen3.6-35B에는 이 외에 Gated DeltaNet 층의 `in_proj_qkv, in_proj_z, out_proj`와 공유 expert가 있고, 256개 expert는 nn.Linear가 아니라 3차원 파라미터라 `target_parameters`(`--lora-experts`)로만 붙는다.

### dropout
LoRA 경로에 넣는 dropout. 데이터가 적으면 0.05~0.1을 쓰기도 한다. train_lora.py는 0 - expert `target_parameters`가 dropout 0을 요구하고, 2,950문항 × 1 epoch라 과적합 여지가 적다.

## 실행하고 볼 것

```bash
uv run steps/07_lora.py
```
- A: `A+B`가 W의 12.5%(r 16), 학습 전 출력 동일, 5 step 뒤 달라짐, W의 grad는 None(얼림).
- B: 층 하나의 Linear 목록. `trainable params: 10,092,544 || all params: 606,142,464 || trainable%: 1.67`. 감싼 뒤 `q_proj`가 PEFT의 `Linear`(안에 lora_A, lora_B)로 바뀜. 출력 동일.
- C: 어댑터 파일 40MB(0.6B, r 16). 원래 가중치 1.2GB는 저장하지 않는다.

## 실험해 보기 (실제 돌린 결과)

| 명령 | trainable params | 어댑터 크기 |
|---|---|---|
| `--r 4` | 2.5M (0.42%) | 10MB |
| `--r 16` (기본) | 10.1M (1.67%) | 40MB |
| `--r 64` | 40.4M (6.3%) | 162MB |
| `--targets q_proj,v_proj` (r 16) | 2.3M (0.38%) | 9MB |

**경향**: 파라미터 수는 r × (붙인 모듈의 입력+출력 차원 합)에 비례. 품질에 미치는 영향은 8단계에서 직접 잰다.

## train_lora.py와의 연결
```python
lora = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=config.LORA_TARGET_MODULES, target_parameters=... if args.lora_experts else None)
model = get_peft_model(model, lora)                      # B
model.print_trainable_parameters()                       # B
params = [p for p in model.parameters() if p.requires_grad]   # A·B만 optimizer에 (얼린 W는 제외)
...
model.save_pretrained(args.out)                          # C: 어댑터만 저장
```
`task_type="CAUSAL_LM"`: PEFT에게 "다음 토큰 예측 모델"이라고 알려 준다(저장 메타데이터와 generate 호환용).
