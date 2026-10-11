"""Turn coding-agent trajectories into fixed-length token sequences for the eviction evals.

    python scripts/prepare_agent_data.py --model deepseek-ai/deepseek-coder-1.3b-instruct \
        --seq-len 8192 --n-seqs 96 --out data/agent_8k.pt

Source: nebius/SWE-rebench-openhands-trajectories (OpenHands sessions on real GitHub issues,
CC-BY-4.0). Each session is rendered as plain text, one block per message:

    <|system|> ... <|assistant|> reasoning and tool calls ... <|tool|> tool output ...

and cut to its first `seq_len` tokens; shorter sessions are skipped. The saved file holds
    ids   [n, seq_len] token ids (BOS first)
    mask  [n, seq_len] True on tokens the agent itself wrote (assistant messages)
    meta  per-sequence trajectory id, resolved flag, original length, assistant share
The script also prints how repetitive the text is (see `repeat_share`), next to WikiText.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from subcache.evals import load_text  # noqa: E402


def message_text(msg):
    """Readable text of one message, tool calls included."""
    content = msg.get("content")
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    parts = [content or ""]
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", call) if isinstance(call, dict) else {}
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args)
        parts.append(f"CALL {fn.get('name', '')} {args}")
    return "\n".join(p for p in parts if p)


def render(tok, trajectory, seq_len, offset=0):
    """Token ids and assistant mask for one trajectory, or None if it is too short.

    offset > 0 takes a window from later in the session: it starts at the first message
    boundary at or after `offset` tokens, with a BOS token in front.
    """
    ids, mask, starts = [tok.bos_token_id], [False], []
    for msg in trajectory:
        role = msg.get("role", "user")
        piece = tok(f"<|{role}|>\n{message_text(msg)}\n", add_special_tokens=False)["input_ids"]
        starts.append(len(ids))
        ids += piece
        mask += [role == "assistant"] * len(piece)
    total = len(ids)
    if offset:
        begin = next((st for st in starts if st >= offset), None)
        if begin is None:
            return None
        ids, mask = [tok.bos_token_id] + ids[begin:], [False] + mask[begin:]
    if len(ids) < seq_len:
        return None
    return ids[:seq_len], mask[:seq_len], total


def repeat_share(ids, n=16):
    """Share of positions where the n tokens ending there already occurred earlier in the same
    sequence: a direct measure of verbatim repetition."""
    seen, repeats = set(), 0
    ids = list(ids)
    for end in range(n, len(ids) + 1):
        gram = tuple(ids[end - n:end])
        repeats += gram in seen
        seen.add(gram)
    return repeats / max(1, len(ids) - n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    ap.add_argument("--dataset", default="nebius/SWE-rebench-openhands-trajectories")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--n-seqs", type=int, default=96)
    ap.add_argument("--offset", type=int, default=0, help="start this many tokens into each session (0 = from the start)")
    ap.add_argument("--max-scan", type=int, default=3000, help="stop after reading this many trajectories")
    ap.add_argument("--out", default="data/agent_8k.pt")
    args = ap.parse_args()

    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load_dataset(args.dataset, split="train", streaming=True)
    ids, masks, meta, seen_repos = [], [], [], {}
    for scanned, row in enumerate(rows):
        if len(ids) >= args.n_seqs or scanned >= args.max_scan:
            break
        if seen_repos.get(row["repo"], 0) >= 2:  # spread over repositories
            continue
        out = render(tok, row["trajectory"], args.seq_len, args.offset)
        if out is None:
            continue
        seen_repos[row["repo"]] = seen_repos.get(row["repo"], 0) + 1
        ids.append(out[0])
        masks.append(out[1])
        meta.append({"trajectory_id": row["trajectory_id"], "repo": row["repo"], "resolved": int(row["resolved"]),
                     "full_tokens": out[2], "assistant_share": sum(out[1]) / len(out[1])})
    if len(ids) < args.n_seqs:
        raise SystemExit(f"only {len(ids)} trajectories of >= {args.seq_len} tokens in the first {args.max_scan}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"ids": torch.tensor(ids), "mask": torch.tensor(masks), "meta": meta,
                "model": args.model, "dataset": args.dataset}, args.out)

    agent_rep = [repeat_share(x) for x in ids[:32]]
    wiki = tok(load_text("wikitext"), add_special_tokens=False)["input_ids"]
    wiki_rep = [repeat_share(wiki[i * args.seq_len:(i + 1) * args.seq_len]) for i in range(min(32, len(wiki) // args.seq_len))]
    summary = {
        "sequences": len(ids), "seq_len": args.seq_len, "offset": args.offset, "repos": len(seen_repos),
        "resolved_share": sum(m["resolved"] for m in meta) / len(meta),
        "assistant_token_share": sum(m["assistant_share"] for m in meta) / len(meta),
        "median_full_tokens": sorted(m["full_tokens"] for m in meta)[len(meta) // 2],
        "repeat_share_agent": sum(agent_rep) / len(agent_rep),
        "repeat_share_wikitext": sum(wiki_rep) / max(1, len(wiki_rep)),
    }
    print(json.dumps(summary, indent=1))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("sample:\n" + tok.decode(ids[0][:400]))


if __name__ == "__main__":
    main()
