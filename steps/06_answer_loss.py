"""6단계: 답 토큰에만 loss - 답 토큰 loss 함수 answer_loss() 를 조각내서 확인한다.

    n_answer = ids.shape[1] - p_len
    out = model(input_ids=ids, logits_to_keep=n_answer + 1, use_cache=False)   # 마지막 A+1 위치의 logits만
    logits = out.logits[0, :-1].float()                                        # 위치 P-1 .. L-2  (각각 토큰 P .. L-1을 예측)
    targets = ids[0, p_len:]                                                   # 토큰 P .. L-1
    ce = F.cross_entropy(logits, targets, reduction="none")
    weight = where(targets == y_id, y_weight, 1.0)
    loss = (ce * weight).sum() / weight.sum()

실행:  uv run steps/06_answer_loss.py
       uv run steps/06_answer_loss.py --y-weight 3
       uv run steps/06_answer_loss.py --mem-demo --seq-len 1024     # gradient checkpointing의 메모리 효과(시간이 좀 걸린다)
"""

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MESSAGES = [
    {"role": "system", "content": "각 문서가 질문에 필요한지 Y/N으로 답하라."},
    {"role": "user", "content": "질문: 재고 수량이 안 맞아요\n\n문서 후보:\n[1] 재고 조정 메뉴\n[2] 급여 명세서 출력\n[3] 재고 실사 입력"},
    {"role": "assistant", "content": "[1] Y\n[2] N\n[3] Y"},
]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def encode(tok, messages):  # 5단계와 같다
    kw = dict(tokenize=True, enable_thinking=False, return_dict=False)
    full = tok.apply_chat_template(messages, **kw)
    prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True, **kw)
    assert full[: len(prompt)] == prompt
    end = max(i for i, t in enumerate(full) if t == tok.convert_tokens_to_ids("<|im_end|>"))
    return full[: end + 1], len(prompt)


def part_a_two_ways(model, ids, p_len) -> torch.Tensor:
    print("=" * 70)
    print("A. 같은 loss를 두 방법으로: (1) 전체 logits를 구해 답 부분만 자르기  (2) logits_to_keep로 답 부분만 계산하기")
    n_answer = ids.shape[1] - p_len
    with torch.no_grad():
        full_logits = model(input_ids=ids, use_cache=False).logits  # [1, L, 어휘]
        kept = model(input_ids=ids, logits_to_keep=n_answer + 1, use_cache=False).logits  # [1, A+1, 어휘]
    targets = ids[0, p_len:]
    loss1 = F.cross_entropy(full_logits[0, p_len - 1 : -1].float(), targets)  # 위치 P-1..L-2 가 토큰 P..L-1 을 예측
    loss2 = F.cross_entropy(kept[0, :-1].float(), targets)
    vocab = full_logits.shape[-1]
    print(f"  L = {ids.shape[1]} (P {p_len} + A {n_answer})")
    print(f"  (1) 전체 logits {tuple(full_logits.shape)} = {full_logits.numel() * 2 / 1e6:6.1f}MB(bf16)  -> loss {loss1.item():.4f}")
    print(f"  (2) kept  logits {tuple(kept.shape)} = {kept.numel() * 2 / 1e6:6.1f}MB(bf16)  -> loss {loss2.item():.4f}   <- 같다")
    print(f"  실제 데이터(L 13,000, 어휘 248,320, fp32로 계산)라면 (1)은 {13000 * 248320 * 4 / 1e9:.1f}GB, (2)는 {53 * 248320 * 4 / 1e6:.0f}MB")
    print("  logits_to_keep=A+1 인 이유: 토큰 P를 예측하는 것은 위치 P-1의 logits -> 마지막 A+1개가 필요하고, 그중 맨 끝(위치 L-1)은 정답이 없어 버린다([:-1])")
    return kept[0, :-1].float()


def part_b_per_token(tok, logits, ids, p_len, y_id, y_weight) -> None:
    print("\n" + "=" * 70)
    print(f"B. 답 토큰마다 -log P(정답), 그리고 ' Y' 가중치 {y_weight}")
    targets = ids[0, p_len:]
    ce = F.cross_entropy(logits, targets, reduction="none")
    weight = torch.where(targets == y_id, torch.tensor(y_weight, device=ce.device), torch.tensor(1.0, device=ce.device))
    probs = torch.softmax(logits, dim=-1)
    print(f"  {'토큰':14s} {'P(정답)':>8s} {'-log P':>7s} {'가중치':>5s}   모델이 1등으로 꼽은 토큰")
    for t, c, w, p in zip(targets.tolist(), ce.tolist(), weight.tolist(), probs):
        print(f"  {tok.decode([t])!r:14s} {p[t].item():8.3f} {c:7.3f} {w:5.1f}   {tok.decode([p.argmax().item()])!r}")
    plain = ce.mean().item()
    weighted = ((ce * weight).sum() / weight.sum()).item()
    print(f"  가중치 없는 평균 = {plain:.4f},  가중 평균 (ce*weight).sum()/weight.sum() = {weighted:.4f}")
    print("  형식 토큰('[', 숫자, ']', '\\n')은 학습 전에도 대부분 쉽고(두 번째 줄부터), ' Y'/' N'과 <|im_end|>가 어렵다 - 학습이 실제로 바꾸는 부분")


def mem_now(device: str) -> float:
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def part_c_mem_demo(model, tok, device: str, seq_len: int, y_id) -> None:
    print("\n" + "=" * 70)
    print(f"C. gradient checkpointing: 길이 {seq_len} 입력으로 forward 뒤 '남아 있는 활성값' 메모리를 비교 (backward에 쓰려고 저장해 둔 것)")
    text = "재고 수량이 안 맞아요. " * (seq_len // 8)
    ids = torch.tensor(tok.encode(text, add_special_tokens=False)[: seq_len - 8] + tok.encode("[1] Y\n[2] N", add_special_tokens=False), device=device)[None]
    p_len = ids.shape[1] - 9
    model.enable_input_require_grads()  # 입력 임베딩부터 기울기가 흐르게 -> 모든 층의 활성값이 저장 대상이 된다 (실전 설정과 같다)
    for ckpt in (False, True):
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        model.train()
        base = mem_now(device)
        t0 = time.time()
        n_answer = ids.shape[1] - p_len
        out = model(input_ids=ids, logits_to_keep=n_answer + 1, use_cache=False)
        loss = F.cross_entropy(out.logits[0, :-1].float(), ids[0, p_len:])
        after_fwd = mem_now(device)
        loss.backward()
        t = time.time() - t0
        model.zero_grad(set_to_none=True)
        del out, loss
        print(f"  checkpointing {'켬 ' if ckpt else '끔 '}: forward 뒤 추가 메모리 {after_fwd - base:5.2f}GB, forward+backward {t:5.1f}초")
    print("  켜면 활성값을 저장하지 않고 backward 때 다시 계산한다 -> 메모리 ↓, 시간 ↑(약 30%). 35B에 13k 토큰을 넣으려면 켜야 한다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--y-weight", type=float, default=1.0)
    ap.add_argument("--mem-demo", action="store_true")
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    y_id = tok.encode(" Y", add_special_tokens=False)[0]

    full, p_len = encode(tok, MESSAGES)
    ids = torch.tensor(full, device=device)[None]
    logits = part_a_two_ways(model, ids, p_len)
    part_b_per_token(tok, logits, ids, p_len, y_id, args.y_weight)
    if args.mem_demo:
        part_c_mem_demo(model, tok, device, args.seq_len, y_id)


if __name__ == "__main__":
    main()
