"""8단계: 작은 SFT 실험실 - 1~7단계의 조각을 하나의 학습 루프로 조립해 "미니 필터"를 학습하고, 파라미터를 바꿔 가며 결과를 비교한다.

작업: 질문 1개 + 짧은 문서 4개 -> "[1] N\\n[2] Y\\n[3] N\\n[4] N" (실전 필터 데이터의 축소판. 문서가 한 줄이라 맥에서 몇 분이면 돈다).
val은 train에 없는 주제로만 만든다(실전에서도 val과 train의 문서 출처가 겹치지 않게 나눈다).

학습 루프·loss·encode는 실전 학습 스크립트(9단계)와 같다(이름도 같다). 다른 것은 데이터가 작고, 끝나면 바로 평가한다는 것뿐.

실행
  uv run steps/08_sft_lab.py train --tag base_lr1e-4                       # 기본값(lr 1e-4, r 16, 40 step)
  uv run steps/08_sft_lab.py train --tag lr1e-3 --lr 1e-3
  uv run steps/08_sft_lab.py train --tag yw3 --y-weight 3
  uv run steps/08_sft_lab.py plot base_lr1e-4 lr1e-3 yw3                   # loss 곡선 겹쳐 그리기 -> runs/compare.png
결과: runs/<tag>/{train_log.jsonl, eval.json, adapter/}
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

RUNS = Path(__file__).resolve().parent.parent / "runs"
SYSTEM_PROMPT = "질문에 답하는 데 필요한 문서인지 문서마다 Y/N으로 답하라. 형식: [번호] Y 또는 [번호] N, 한 줄에 하나."

# 주제마다 질문 말투 2개 + 관련 문서 2개. 질문과 문서의 낱말이 조금 다르게(동의어) 해서 글자 맞추기만으로는 못 풀게 한다
TOPICS = [
    ("재고 수량이 실제랑 달라요", "전산 재고랑 창고 재고가 안 맞는데 어떻게 하죠", "재고조정: 전산 재고와 실물 재고의 차이를 맞추는 메뉴입니다.", "재고수불부에서 품목별 입출고 내역과 현재고를 확인할 수 있습니다."),
    ("세금계산서는 어디서 끊나요", "전자세금계산서 발행 방법 알려주세요", "전자세금계산서 발행: 매출전표를 선택하고 국세청 전송을 누릅니다.", "세금계산서 발행 전 거래처의 사업자번호와 이메일을 등록해야 합니다."),
    ("급여명세서 뽑는 법", "직원들 월급 명세서 출력은 어떻게 해요", "급여명세서 출력: 급여관리 > 명세서에서 월을 선택해 인쇄합니다.", "급여명세서는 PDF로 저장하거나 직원 이메일로 일괄 발송할 수 있습니다."),
    ("새 거래처 추가하고 싶어요", "거래처 신규 등록은 어디서 하나요", "거래처등록: 기초정보 > 거래처에서 사업자번호와 상호를 입력합니다.", "거래처 정보는 엑셀 양식으로 여러 건을 한 번에 등록할 수 있습니다."),
    ("비밀번호 바꾸려면", "로그인 암호 변경 메뉴가 어디예요", "비밀번호 변경: 내 정보 > 보안에서 현재 암호와 새 암호를 입력합니다.", "비밀번호는 8자 이상, 영문과 숫자를 섞어야 합니다."),
    ("품목에 바코드 붙이는 방법", "상품 바코드 등록 어떻게 하죠", "품목등록의 바코드 항목에 코드를 입력하면 스캐너로 입출고할 수 있습니다.", "바코드 라벨은 품목 목록에서 선택 후 라벨 인쇄로 출력합니다."),
    ("지출결의서 올렸는데 결재가 안 넘어가요", "지출 결의 승인 요청은 어떻게 하나요", "지출결의서 작성 후 상신을 눌러야 결재선의 승인자에게 전달됩니다.", "결재 진행 상태는 전자결재 > 상신함에서 확인합니다."),
    ("판매 입력한 걸 취소하고 싶어요", "잘못 넣은 판매 전표 지우는 법", "판매입력 조회에서 해당 전표를 선택하고 삭제를 누르면 취소됩니다.", "판매 전표를 삭제하면 연결된 재고 출고도 함께 취소됩니다."),
    ("외상으로 판 돈 얼마나 남았는지 보려면", "미수금 현황 어디서 확인해요", "외상매출금 현황: 거래처별 미수 잔액과 회수 예정일을 보여 줍니다.", "미수금은 입금 전표를 입력하면 자동으로 차감됩니다."),
    ("직원별로 메뉴 권한 다르게 주고 싶어요", "사용자 권한 설정은 어떻게 하나요", "사용자권한: 사용자별로 메뉴 조회·입력·삭제 권한을 지정합니다.", "권한 그룹을 만들어 여러 사용자에게 같은 권한을 한 번에 줄 수 있습니다."),
    ("전표 하나 지우려면", "입력한 회계 전표 삭제 방법", "전표 조회에서 전표를 선택하고 삭제하면 됩니다. 마감된 기간은 삭제할 수 없습니다.", "삭제한 전표는 삭제 이력에서 번호와 삭제자를 확인할 수 있습니다."),
    ("로그인이 계속 안 돼요", "계정이 잠겼다고 나와요", "로그인 5회 실패 시 계정이 잠기며 관리자가 잠금 해제할 수 있습니다.", "잠금 해제는 사용자 관리에서 해당 계정을 선택하고 해제를 누릅니다."),
    ("품목 여러 개를 엑셀로 한 번에 넣고 싶어요", "품목 일괄 등록 방법", "품목 엑셀 업로드: 양식을 내려받아 채운 뒤 업로드하면 일괄 등록됩니다.", "엑셀 업로드 시 품목코드가 중복되면 해당 행은 건너뜁니다."),
    ("견적서 양식 바꾸고 싶어요", "출력물 서식 수정은 어디서", "출력양식 설정: 견적서·거래명세서의 항목과 로고를 편집합니다.", "출력양식은 여러 개 저장해 두고 인쇄할 때 고를 수 있습니다."),
    ("재고 실사는 어떻게 하죠", "창고 재고 세어서 반영하는 법", "재고실사: 실사 수량을 입력하면 전산 재고와의 차이가 자동 계산됩니다.", "실사 결과는 재고조정 전표로 반영할 수 있습니다."),
    ("달러 거래 환율 설정", "외화 환율은 어디서 바꿔요", "환율등록: 통화별 기준 환율을 날짜별로 등록합니다.", "외화 전표는 입력 당일의 등록 환율로 원화 환산됩니다."),
    ("휴가 신청은 어디서 해요", "연차 쓰려면 어떻게 하죠", "근태 > 휴가신청에서 기간과 사유를 입력하고 상신합니다.", "휴가 승인 후 남은 연차는 근태 현황에서 확인합니다."),
    ("매입 전표 입력하는 법", "구매한 거 전표로 넣으려면", "매입입력: 거래처, 품목, 수량, 단가를 입력하면 매입 전표와 입고가 생성됩니다.", "매입 전표에서 세금계산서를 수취 처리하면 부가세 자료에 반영됩니다."),
    ("창고를 하나 더 만들고 싶어요", "새 창고 추가 방법", "창고등록: 기초정보 > 창고에서 창고 코드와 이름을 입력합니다.", "창고별로 재고를 따로 관리하고 창고 간 이동 전표를 입력할 수 있습니다."),
    ("알림 메일이 안 와요", "메일 알림 설정 어디서 하나요", "알림설정: 전자결재·일정 알림을 메일 또는 메시지로 받을지 선택합니다.", "메일 알림은 사용자 정보에 이메일이 등록되어 있어야 발송됩니다."),
    ("부가세 신고 자료 뽑는 법", "부가가치세 신고용 자료 어디서 받아요", "부가세신고: 기간을 선택하면 매출·매입 세금계산서 합계표가 만들어집니다.", "부가세 신고 자료는 홈택스 업로드용 파일로 내려받을 수 있습니다."),
    ("품목 단가 한꺼번에 바꾸고 싶어요", "판매가 일괄 수정 방법", "단가 일괄변경: 품목을 선택하고 비율 또는 금액으로 단가를 바꿉니다.", "거래처별 단가는 거래처 단가표에서 따로 관리합니다."),
    ("결재선은 어떻게 정해요", "승인자 순서 설정 방법", "결재선 설정: 문서 종류별로 승인자 순서를 등록합니다.", "결재선은 부서별로 기본값을 두고 상신할 때 바꿀 수 있습니다."),
    ("데이터 백업은 어떻게 받아요", "자료 내보내기 방법", "데이터 내보내기: 메뉴별 자료를 엑셀로 내려받습니다.", "전체 백업은 관리자 메뉴의 자료 백업에서 요청합니다."),
]
N_DOCS = 4


# ---------------------------------------------------------------- 데이터

def make_rows(topic_ids: list[int], n: int, rng: random.Random) -> list[dict]:
    """예시 n개: 주제 하나의 질문 + (그 주제 문서 1~2개 + 다른 주제 문서)를 섞어 4개. labels는 문서마다 Y/N."""
    rows = []
    for _ in range(n):
        t = rng.choice(topic_ids)
        q = rng.choice(TOPICS[t][:2])
        n_gold = 1 if rng.random() < 0.7 else 2
        gold = rng.sample(TOPICS[t][2:], n_gold)
        others = [d for u in topic_ids if u != t for d in TOPICS[u][2:]]
        docs = [(d, True) for d in gold] + [(d, False) for d in rng.sample(others, N_DOCS - n_gold)]
        rng.shuffle(docs)
        user = f"질문: {q}\n\n문서 후보:\n" + "\n".join(f"[{i}] {d}" for i, (d, _) in enumerate(docs, 1))
        target = "\n".join(f"[{i}] {'Y' if g else 'N'}" for i, (_, g) in enumerate(docs, 1))
        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}, {"role": "assistant", "content": target}],
                     "labels": [g for _, g in docs]})
    return rows


def make_data(n_train: int, n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    ids = list(range(len(TOPICS)))
    rng.shuffle(ids)
    val_topics, train_topics = ids[:4], ids[4:]  # val 주제 4개는 train에 없다
    return make_rows(train_topics, n_train, rng), make_rows(val_topics, n_val, rng)


# ---------------------------------------------------------------- 실전 학습 스크립트와 같은 부분 (5, 6단계)

def encode(tok, messages: list[dict]) -> tuple[list[int], int]:
    kw = dict(tokenize=True, enable_thinking=False, return_dict=False)
    full = tok.apply_chat_template(messages, **kw)
    prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True, **kw)
    assert full[: len(prompt)] == prompt
    end = max(i for i, t in enumerate(full) if t == tok.convert_tokens_to_ids("<|im_end|>"))
    return full[: end + 1], len(prompt)


def answer_loss(model, ids: torch.Tensor, p_len: int, y_id: int, y_weight: float) -> torch.Tensor:
    n_answer = ids.shape[1] - p_len
    out = model(input_ids=ids, logits_to_keep=n_answer + 1, use_cache=False)
    logits = out.logits[0, :-1].float()
    targets = ids[0, p_len:]
    ce = F.cross_entropy(logits, targets, reduction="none")
    weight = torch.where(targets == y_id, torch.tensor(y_weight, device=ce.device), torch.tensor(1.0, device=ce.device))
    return (ce * weight).sum() / weight.sum()


# ---------------------------------------------------------------- 평가 (실전 평가 스크립트의 점수 매기기 + 지표 계산을 합친 것)

def doc_probs(tok, gen_ids, step_logits, y_ids, n_ids, n_docs) -> list[float] | None:
    probs = []
    for t, logits in zip(gen_ids, step_logits):
        if t in y_ids | n_ids:
            p = torch.softmax(logits[0].float(), dim=-1)
            py, pn = sum(p[i].item() for i in y_ids), sum(p[i].item() for i in n_ids)
            probs.append(py / (py + pn) if py + pn > 0 else 0.0)
    numbers = [int(m) for m in re.findall(r"\[(\d+)\]\s*[YN]", tok.decode(gen_ids, skip_special_tokens=True))]
    return probs if numbers == list(range(1, n_docs + 1)) and len(probs) == n_docs else None


@torch.no_grad()
def evaluate(model, tok, rows: list[dict], device: str) -> dict:
    """greedy 생성 -> 형식 준수율, 문서 단위 정확도/precision/recall(Y 기준), 문항 단위 전부 일치, 라벨별 평균 P(Y)."""
    model.eval()
    y_ids = {tok.encode(s, add_special_tokens=False)[0] for s in (" Y", "Y")}
    n_ids = {tok.encode(s, add_special_tokens=False)[0] for s in (" N", "N")}
    ok, exact, tp, fp, fn, py_y, py_n, sample = 0, 0, 0, 0, 0, [], [], None
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:2], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids = torch.tensor(prompt, device=device)[None]
        out = model.generate(ids, max_new_tokens=48, do_sample=False, output_logits=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        gen = out.sequences[0, ids.shape[1]:].tolist()
        probs = doc_probs(tok, gen, out.logits, y_ids, n_ids, N_DOCS)
        if sample is None:
            sample = tok.decode(gen, skip_special_tokens=True)
        if probs is None:
            fn += sum(r["labels"])  # 형식이 깨지면 그 문항의 Y는 전부 놓친 것으로
            continue
        ok += 1
        picked = [p >= 0.5 for p in probs]  # greedy가 낸 글자와 같다
        exact += picked == r["labels"]
        for p, g, pr in zip(picked, r["labels"], probs):
            tp += p and g
            fp += p and not g
            fn += (not p) and g
            (py_y if g else py_n).append(pr)
    n = len(rows)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"n": n, "format_ok": ok / n, "exact": exact / n, "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "mean_py_on_Y": sum(py_y) / max(1, len(py_y)), "mean_py_on_N": sum(py_n) / max(1, len(py_n)), "sample_output": sample}


def show(name: str, m: dict) -> None:
    print(f"  {name:8s} 형식 {m['format_ok']:.2f} | 문항 전부 일치 {m['exact']:.2f} | Y precision {m['precision']:.2f} recall {m['recall']:.2f} F1 {m['f1']:.2f} "
          f"| 평균 P(Y): Y라벨 {m['mean_py_on_Y']:.2f} / N라벨 {m['mean_py_on_N']:.2f} | 첫 문항 출력 {m['sample_output']!r}")


# ---------------------------------------------------------------- 학습

def train(args) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    y_id = tok.encode(" Y", add_special_tokens=False)[0]

    train_rows, val_rows = make_data(args.n_train, args.n_val, args.seed)
    examples = [encode(tok, r["messages"]) for r in train_rows]
    n_y = sum(sum(r["labels"]) for r in train_rows)
    print(f"train {len(train_rows)}문항 (Y 라벨 {n_y / (len(train_rows) * N_DOCS) * 100:.0f}%), val {len(val_rows)}문항(다른 주제) | 토큰 길이 중앙값 {sorted(len(e[0]) for e in examples)[len(examples) // 2]} | device {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM", target_modules=args.targets.split(","))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("학습 전 val:")
    base = evaluate(model, tok, val_rows, device)
    show("base", base)

    params = [p for p in model.parameters() if p.requires_grad]
    total_steps = args.steps
    warmup = max(1, int(total_steps * args.warmup_ratio))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    print(f"optimizer step {total_steps}개 (warmup {warmup}), accum {args.accum}, lr {args.lr}, r {args.r}, alpha {args.alpha}, y_weight {args.y_weight}")

    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    model.train()
    log = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")
    t0, losses, opt_step, i = time.time(), [], 0, 0
    while opt_step < total_steps:
        ids, p_len = examples[order[i % len(order)]]  # 예시가 모자라면 처음부터 다시(여러 epoch)
        i += 1
        ids = torch.tensor(ids, device=device)[None]
        loss = answer_loss(model, ids, p_len, y_id, args.y_weight)
        (loss / args.accum).backward()
        losses.append(loss.item())
        if i % args.accum == 0:
            norm = torch.nn.utils.clip_grad_norm_(params, args.clip if args.clip > 0 else float("inf"))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            rec = {"step": opt_step, "loss": sum(losses[-args.accum:]) / args.accum, "lr": sched.get_last_lr()[0], "grad_norm": norm.item(), "epoch": i / len(order)}
            if opt_step <= 3 or opt_step % 5 == 0 or opt_step == total_steps:
                print(json.dumps(rec, ensure_ascii=False))
            log.write(json.dumps(rec) + "\n")
    elapsed = time.time() - t0
    model.save_pretrained(out_dir / "adapter")

    print(f"학습 {elapsed / 60:.1f}분. 학습 후 val:")
    after = evaluate(model, tok, val_rows, device)
    show("lora", after)
    (out_dir / "eval.json").write_text(json.dumps({"args": vars(args), "base": base, "lora": after, "train_min": elapsed / 60}, ensure_ascii=False, indent=1))
    print(f"저장: {out_dir}")


def plot(tags: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for tag in tags:
        recs = [json.loads(l) for l in (RUNS / tag / "train_log.jsonl").open()]
        ev = json.loads((RUNS / tag / "eval.json").read_text())
        ax.plot([r["step"] for r in recs], [r["loss"] for r in recs], label=f"{tag} (val F1 {ev['base']['f1']:.2f} -> {ev['lora']['f1']:.2f})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss (answer-token cross-entropy)")  # 기본 글꼴에 한글이 없어 영문으로
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUNS / "compare.png", dpi=120)
    print(f"저장: {RUNS / 'compare.png'}")
    print(f"  {'tag':16s} {'형식':>5s} {'전부일치':>6s} {'P':>5s} {'R':>5s} {'F1':>5s} {'P(Y)|Y':>7s} {'P(Y)|N':>7s}  (학습 전 -> 후)")
    for tag in tags:
        ev = json.loads((RUNS / tag / "eval.json").read_text())
        b, a = ev["base"], ev["lora"]
        print(f"  {tag:16s} {b['format_ok']:.2f}>{a['format_ok']:.2f} {b['exact']:.2f}>{a['exact']:.2f} {b['precision']:.2f}>{a['precision']:.2f} {b['recall']:.2f}>{a['recall']:.2f} "
              f"{b['f1']:.2f}>{a['f1']:.2f} {b['mean_py_on_Y']:.2f}>{a['mean_py_on_Y']:.2f} {b['mean_py_on_N']:.2f}>{a['mean_py_on_N']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True, help="runs/<tag>/ 에 저장")
    t.add_argument("--model", default="Qwen/Qwen3-0.6B")
    t.add_argument("--n-train", type=int, default=200)
    t.add_argument("--n-val", type=int, default=40)
    t.add_argument("--steps", type=int, default=40, help="optimizer step 수")
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    t.add_argument("--y-weight", type=float, default=1.0)
    t.add_argument("--warmup-ratio", type=float, default=0.03)
    t.add_argument("--clip", type=float, default=1.0, help="0이면 끔")
    t.add_argument("--checkpointing", action="store_true", help="gradient checkpointing(메모리 ↓ 시간 ↑). 맥에서는 필요 없다")
    t.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plot")
    p.add_argument("tags", nargs="+")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else plot(args.tags)


if __name__ == "__main__":
    main()
