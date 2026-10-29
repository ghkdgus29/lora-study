# 용어집 (train_lora.py · eval_filter.py에 나오는 순서대로)

| 용어 | 뜻 | 단계 |
|---|---|---|
| 텐서 (tensor) | 숫자 배열. 파라미터·입력·loss 전부 텐서 | 1 |
| 파라미터 (parameter) | 학습으로 고치는 숫자. 35B = 350억 개 | 1 |
| loss (손실) | "얼마나 틀렸나"를 숫자 하나로. 학습 = 이것을 줄이기 | 1 |
| 기울기 (gradient, `.grad`) | 파라미터를 조금 늘리면 loss가 얼마나 변하나(미분값) | 1 |
| autograd / `backward()` | 계산 과정을 거꾸로 따라가 모든 파라미터의 기울기를 자동으로 구하는 것(역전파) | 1 |
| 학습률 (lr, learning rate) | 한 step에 기울기 반대 방향으로 가는 보폭 | 1, 2 |
| optimizer / `step()` | 기울기로 파라미터를 고치는 규칙. SGD(그대로), AdamW(기울기 크기를 정규화) | 2 |
| `zero_grad()` | 누적된 기울기 비우기. backward는 grad를 더하기 때문 | 1, 2 |
| 배치 (batch) | 한 번에 같이 재는 예시 묶음. 클수록 기울기가 안정 | 2 |
| gradient accumulation (`accum`) | 작은 묶음으로 backward를 여러 번 해 큰 배치와 같은 기울기를 만드는 것 | 2 |
| warmup | 처음 몇 step 동안 lr을 0에서 서서히 올리는 것 | 2 |
| 스케줄 (scheduler, `LambdaLR`) | step마다 lr을 바꾸는 규칙(warmup + 선형 감소) | 2 |
| grad clipping (`clip_grad_norm_`) | 기울기 벡터 길이가 한도를 넘으면 줄이는 안전장치 | 2 |
| epoch | 학습 데이터 전체를 한 바퀴 | 2, 8 |
| logits | 모델이 클래스(토큰)마다 낸 점수. 아직 확률 아님 | 3 |
| softmax | 점수를 합이 1인 확률로 | 3 |
| cross-entropy | -log P(정답). 분류의 표준 loss | 3 |
| 가중 평균 / `y_weight` | 특정 토큰(' Y')의 loss를 몇 배로 세는 것. 라벨 불균형 대응 | 3 |
| precision / recall / F1 | Y로 고른 것 중 정답 비율 / 정답 중 고른 비율 / 둘의 조화 평균 | 3 |
| 임계값 (threshold) | P(Y)가 이 값 이상이면 Y로 판정. greedy는 0.5 | 3 |
| 토크나이저 (tokenizer) | 글자 ↔ 정수 id. `' Y'`와 `'Y'`는 다른 토큰 | 4 |
| 어휘 (vocab) | 토큰의 전체 집합(Qwen3.6: 248,320) = 분류의 클래스 수 | 4 |
| forward | 입력을 모델에 통과시켜 logits를 얻는 것 | 4 |
| greedy 생성 | 매 step 확률 최대 토큰을 고르는 것(`do_sample=False`) | 4 |
| teacher forcing | 정답 문장을 통째로 넣고 모든 위치를 한 번에 채점하는 학습 방식 | 4, 6 |
| in-context learning | 학습 없이 문맥의 패턴을 따라 하는 것 | 4 |
| chat template | system/user/assistant 메시지를 `<\|im_start\|>…<\|im_end\|>` 텍스트로 펼치는 규칙 | 5 |
| `enable_thinking` | Qwen3의 생각(`<think>`) 블록 사용 여부. 서비스는 끔 | 5 |
| `add_generation_prompt` | "여기서부터 답을 써라" 접두사(`<\|im_start\|>assistant\n…`)를 붙이기 | 5 |
| eos (`<\|im_end\|>`) | 생성 종료 토큰 | 5 |
| P / A (프롬프트 / 답) | loss를 안 거는 앞부분 / 거는 뒷부분 | 5, 6 |
| SFT (supervised fine-tuning) | (입력, 정답 출력) 쌍으로 답 토큰에만 loss를 걸어 학습 | 5, 6 |
| `logits_to_keep` | 마지막 n개 위치의 logits만 계산(메모리 절약) | 6 |
| 활성값 (activation) | backward에 쓰려고 저장해 두는 각 층의 중간 결과 | 6 |
| gradient checkpointing | 활성값을 저장 않고 backward 때 다시 계산(메모리 ↓ 시간 ↑) | 6 |
| bf16 | 2바이트 실수 형식. 가중치 저장용. 정밀도 낮음 | 1, 6 |
| `use_cache` | 생성용 KV 캐시. 학습에선 끔 | 6 |
| LoRA | 큰 가중치 W를 얼리고 작은 A, B(보정)만 학습 | 7 |
| r (rank) | A, B의 폭. 파라미터 수가 비례 | 7 |
| alpha / scale | 보정에 곱하는 배율 alpha/r. 관례 alpha = 2r | 7 |
| `target_modules` | LoRA를 붙일 nn.Linear 이름 | 7 |
| `target_parameters` | nn.Linear가 아닌 파라미터(MoE expert)에 붙일 때 | 7 |
| 어댑터 (adapter) | 학습된 A, B들을 모은 폴더. `save_pretrained` 결과 | 7 |
| PEFT | Hugging Face의 LoRA 등 "일부만 학습" 라이브러리 | 7 |
| 과적합 (overfitting) | train loss는 내려가는데 val 성능이 안 오르거나 내려감 | 8 |
| MoE / expert | Qwen3.6-35B-A3B: 층마다 expert 256개 중 8개만 쓴다(총 35B, 활성 3B) | 7, 9 |
