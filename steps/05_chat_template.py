"""5단계: chat template - messages(system/user/assistant)가 토큰 시퀀스로 어떻게 펼쳐지고, 어디까지가 프롬프트(P)이고 어디부터가 답(A)인가.

train_lora.py의 encode()를 그대로 따라간다:
    full   = tok.apply_chat_template(messages, tokenize=True, enable_thinking=False)              # system+user+assistant 전체
    prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True, enable_thinking=False)  # 답 직전까지
    assert full[:len(prompt)] == prompt                                                          # prompt는 full의 앞부분
    end = 마지막 <|im_end|> 위치;  return full[:end+1], len(prompt)                               # 뒤의 '\\n'은 버린다

실행:  uv run steps/05_chat_template.py
"""

import argparse

from transformers import AutoTokenizer

MESSAGES = [
    {"role": "system", "content": "각 문서가 질문에 필요한지 Y/N으로 답하라."},
    {"role": "user", "content": "질문: 재고 수량이 안 맞아요\n\n문서 후보:\n[1] 재고 조정 메뉴\n[2] 급여 명세서 출력"},
    {"role": "assistant", "content": "[1] Y\n[2] N"},
]


def part_a_template_text(tok) -> None:
    print("=" * 70)
    print("A. messages -> 문자열. 역할마다 <|im_start|>역할\\n ... <|im_end|> 로 감싼다(ChatML 형식)")
    text = tok.apply_chat_template(MESSAGES, tokenize=False, enable_thinking=False)
    print(text)
    print("-" * 70)
    print("enable_thinking=False -> assistant 턴 앞에 빈 <think>\\n\\n</think>\\n\\n 가 들어간다 = '생각 안 하고 바로 답한다'는 표시.")
    print("서비스 필터도 thinking을 끄고 쓰므로 학습 데이터도 같은 모양으로 만든다(학습과 서비스의 프롬프트 모양이 같아야 한다).")
    text_think = tok.apply_chat_template(MESSAGES[:2], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    print(f"참고: enable_thinking=True 이면 끝이 {text_think[-30:]!r} - 빈 think 블록이 없다(모델이 스스로 생각을 쓰게 둔다)")


def part_b_prefix(tok) -> tuple[list[int], list[int]]:
    print("\n" + "=" * 70)
    print("B. add_generation_prompt=True: messages[:2](system+user)만 넣고 '<|im_start|>assistant\\n<think>...</think>\\n\\n' 까지 붙인다 = 모델이 답을 시작할 자리")
    kw = dict(tokenize=True, enable_thinking=False, return_dict=False)
    full = tok.apply_chat_template(MESSAGES, **kw)
    prompt = tok.apply_chat_template(MESSAGES[:2], add_generation_prompt=True, **kw)
    print(f"  full   = {len(full)} 토큰 (system + user + assistant 답까지)")
    print(f"  prompt = {len(prompt)} 토큰 (답 직전까지)")
    print(f"  full[:len(prompt)] == prompt ?  {full[: len(prompt)] == prompt}   <- 그래서 '앞 P개 = 프롬프트, 그 뒤 = 답'으로 자를 수 있다")
    print("  (이 성질이 깨지는 템플릿도 있어서 train_lora.py는 assert로 확인한다)")
    return full, prompt


def part_c_boundary(tok, full: list[int], prompt: list[int]) -> None:
    print("\n" + "=" * 70)
    print("C. 토큰 하나하나: P(프롬프트, loss 없음) / A(답, loss 있음) / 버림")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    end = max(i for i, t in enumerate(full) if t == im_end)  # 마지막 <|im_end|> = 답의 끝
    p_len = len(prompt)
    for i, t in enumerate(full):
        region = "P" if i < p_len else "A" if i <= end else "버림"
        mark = "  <- 답 시작" if i == p_len else "  <- 답 끝(<|im_end|> = eos, 생성이 여기서 멈춘다)" if i == end else ""
        if i < 6 or p_len - 3 <= i:  # 가운데(user 본문)는 생략
            print(f"  {i:3d} {region:3s} {t:7d} {tok.decode([t])!r}{mark}")
        elif i == 6:
            print("  ...")
    print(f"\n  P = {p_len}, A = {end + 1 - p_len} (답 {end + 1 - p_len - 1}토큰 + <|im_end|>), 버림 = {len(full) - end - 1} (템플릿이 <|im_end|> 뒤에 붙이는 '\\n')")
    print("  train_lora.py encode()의 반환 = (full[:end+1], p_len). 뒤의 '\\n'은 모델이 <|im_end|>에서 멈추므로 배울 필요가 없다")
    print(f"  특수 토큰: <|im_start|>={tok.convert_tokens_to_ids('<|im_start|>')}, <|im_end|>={im_end} (= eos {tok.eos_token_id}), pad={tok.pad_token_id}")


def part_d_why(tok) -> None:
    print("\n" + "=" * 70)
    print("D. 왜 프롬프트에는 loss를 안 거나")
    n_prompt = len(tok.apply_chat_template(MESSAGES[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False))
    n_answer = len(tok.encode(MESSAGES[2]["content"], add_special_tokens=False)) + 1
    print(f"  이 예시: 프롬프트 {n_prompt}토큰 vs 답 {n_answer}토큰. 실제 필터 데이터: 프롬프트 약 8,300토큰 vs 답 52토큰")
    print("  프롬프트(질문·문서 본문)까지 예측하게 하면 loss의 99%가 '매뉴얼 본문 외우기'가 되고 정작 Y/N 판단은 1%도 안 된다.")
    print("  -> 답 토큰에만 loss (SFT의 표준). 모델은 프롬프트를 '읽기만' 하고 답을 '쓰는 법'만 배운다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="토크나이저만 쓴다(모델은 안 올린다)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    part_a_template_text(tok)
    full, prompt = part_b_prefix(tok)
    part_c_boundary(tok, full, prompt)
    part_d_why(tok)


if __name__ == "__main__":
    main()
