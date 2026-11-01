"""4단계: LLM은 "다음 토큰 분류기"다 - 토크나이저, logits, greedy 생성, 그리고 "위치 t-1이 토큰 t를 예측한다".

작은 모델(Qwen/Qwen3-0.6B, 1.4GB)을 맥에 올려 본다. 처음 실행하면 Hugging Face Hub에서 내려받는다(~/.cache/huggingface/hub).
학습 쪽이 쓰는 것:  AutoTokenizer, AutoModelForCausalLM, model(input_ids=...).logits, tok.encode(" Y")
평가 쪽이 쓰는 것:  model.generate(..., output_logits=True), softmax의 P(Y) / (P(Y)+P(N))

실행:  uv run steps/04_next_token.py
       uv run steps/04_next_token.py --prompt "오늘 점심은" --max-new 12
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def part_a_tokenizer(tok) -> None:
    print("=" * 70)
    print("A. 토크나이저: 글자 -> 정수 id. 모델은 정수 id만 본다")
    for text in ("[1] Y\n[2] N", "재고 수량이 안 맞아요"):
        ids = tok.encode(text, add_special_tokens=False)
        print(f"  {text!r}")
        print(f"    -> ids {ids}")
        print(f"    -> 토큰별 {[tok.decode([i]) for i in ids]}")
    print(f"  ' Y'(앞에 공백) = {tok.encode(' Y', add_special_tokens=False)}, 'Y' = {tok.encode('Y', add_special_tokens=False)}  <- 다른 토큰! 답 '[1] Y'의 Y는 공백 붙은 쪽")
    print(f"  어휘 크기 = {len(tok):,}개 (Qwen3.6-35B는 248,077개). 다음 토큰 예측 = 이 개수 중 하나를 고르는 분류")


def part_b_logits(tok, model, prompt: str, device: str) -> None:
    print("\n" + "=" * 70)
    print(f"B. forward 한 번: 프롬프트 {prompt!r} 뒤에 올 토큰의 logits")
    ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device=device)[None]  # [배치 1, 길이 L]
    with torch.no_grad():
        logits = model(input_ids=ids).logits  # [1, L, 어휘]
    print(f"  input_ids.shape = {tuple(ids.shape)}  ->  logits.shape = {tuple(logits.shape)}  (위치마다 어휘 전체의 점수)")
    print(f"  logits 하나가 차지하는 메모리: {logits.numel() * 4 / 1e6:.1f}MB (fp32 기준). 13,000토큰이면 {13000 * logits.shape[-1] * 4 / 1e9:.1f}GB -> 실전 코드가 logits_to_keep를 쓰는 이유(6단계)")
    probs = torch.softmax(logits[0, -1].float(), dim=-1)  # 마지막 위치 = '다음에 올 토큰'의 확률
    top = torch.topk(probs, 5)
    print("  마지막 위치의 상위 5개 토큰:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    {tok.decode([i])!r:12s} P = {p:.3f}")


def part_c_greedy(tok, model, prompt: str, device: str, max_new: int) -> None:
    print("\n" + "=" * 70)
    print(f"C. greedy 생성을 손으로: '가장 확률 높은 토큰을 붙이고 다시 forward'를 {max_new}번")
    ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device=device)[None]
    out = ids
    with torch.no_grad():
        for _ in range(max_new):
            next_id = model(input_ids=out).logits[0, -1].argmax()  # 마지막 위치 logits의 최대 = 확률 최대
            out = torch.cat([out, next_id.view(1, 1)], dim=1)
            if next_id.item() == tok.eos_token_id:
                break
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)  # 라이브러리 함수
    print(f"  손으로:          {tok.decode(out[0, ids.shape[1]:])!r}")
    print(f"  model.generate:  {tok.decode(gen[0, ids.shape[1]:])!r}   <- 같다 (do_sample=False가 greedy)")
    print("  평가 스크립트는 이 generate에 output_logits=True를 줘서 매 step의 logits를 받아 Y/N 자리의 확률을 읽는다")


def part_d_teacher_forcing(tok, model, device: str) -> None:
    print("\n" + "=" * 70)
    print("D. 학습의 시선: 정답 문장을 통째로 넣으면, 위치 t-1의 logits가 토큰 t를 '예측'한 것이다 (한 번의 forward로 모든 위치를 채점)")
    text = "[1] Y\n[2] N\n[3] N"
    ids = torch.tensor(tok.encode(text, add_special_tokens=False), device=device)[None]
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    probs = torch.softmax(logits, dim=-1)
    print(f"  문장 {text!r} ({ids.shape[1]}토큰)")
    print(f"  {'위치':>4s}  {'이 위치 토큰':14s} {'모델이 예측한 다음':16s} {'실제 다음':10s} {'P(실제 다음)':>12s}  -log P")
    for t in range(ids.shape[1] - 1):
        actual = ids[0, t + 1].item()
        pred = probs[t].argmax().item()
        p = probs[t, actual].item()
        print(f"  {t:4d}  {tok.decode([ids[0, t].item()])!r:14s} {tok.decode([pred])!r:16s} {tok.decode([actual])!r:10s} {p:12.3f}  {-torch.log(torch.tensor(p)).item():6.2f}")
    print("  -log P 열의 평균이 이 문장에 대한 cross-entropy loss. 학습은 이 값을 줄인다 = 실제 다음 토큰의 확률을 올린다")
    print("  주의: 마지막 토큰 위치의 logits는 '그 다음'을 예측하는데 정답이 없으니 버린다 -> answer_loss()의 logits[0, :-1] (6단계)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="대한민국의 수도는")
    ap.add_argument("--max-new", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)  # 실전과 같은 읽기 방식(bf16, 한 장치에)
    model.eval()
    print(f"모델 {args.model} | 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | device {device} | dtype {next(model.parameters()).dtype}")

    part_a_tokenizer(tok)
    part_b_logits(tok, model, args.prompt, device)
    part_c_greedy(tok, model, args.prompt, device, args.max_new)
    part_d_teacher_forcing(tok, model, device)


if __name__ == "__main__":
    main()
