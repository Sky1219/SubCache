"""Task data, prompts and scoring for the benchmarks run by scripts/run_bench.py.

LongBench (Bai et al., 2023): prompts, generation lengths, middle truncation and metrics follow
the official repository (THUDM/LongBench, LongBench/config and LongBench/metrics.py, MIT
licence). Data: data.zip from the THUDM/LongBench dataset repo, unpacked under data/longbench.

RULER-style synthetic retrieval (after Hsieh et al., 2024): needle-in-a-haystack variants and
variable tracking, with RULER's templates and its string-match-all score. This is a
reimplementation, not the RULER package: the haystack is WikiText-2 instead of Paul Graham
essays, keys are adjective-noun pairs from a fixed list instead of the wonderwords package, and
each needle reads "One of the special magic number for ..." where RULER writes "numbers". That
wording was used for every reported run, so it is kept to stay comparable with them.

MATH-500 (Lightman et al., 2023; HuggingFaceH4/MATH-500): DeepSeek-R1's recommended prompt and
the last \\boxed{...} of the answer, graded with Math-Verify.
"""

import json
import os
import random
import re
import string
from collections import Counter

# ------------------------------------------------------------------ LongBench
LONGBENCH_PROMPT = {
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
LONGBENCH_MAXGEN = {"qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32, "2wikimqa": 32, "gov_report": 512,
                    "qmsum": 512, "lcc": 64, "repobench-p": 64}
NO_CHAT = ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p")  # LongBench feeds these raw


def load_longbench(task, root="data/longbench/data", limit=None):
    with open(os.path.join(root, f"{task}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def longbench_ids(tok, task, row, max_len):
    """Token ids for one LongBench example: middle-truncate the raw prompt to max_len tokens
    (as LongBench's pred.py does), then apply the chat template unless the task is fed raw."""
    prompt = LONGBENCH_PROMPT[task].format(**row)
    ids = tok(prompt, add_special_tokens=False).input_ids
    if len(ids) > max_len:
        half = max_len // 2
        prompt = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
    if task in NO_CHAT:
        return tok(prompt).input_ids
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
                                   return_dict=False)


def _normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred_tokens, gt_tokens):
    common = Counter(pred_tokens) & Counter(gt_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred_tokens), same / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_f1_score(pred, gt):
    return _f1(_normalize_answer(pred).split(), _normalize_answer(gt).split())


def rouge_score(pred, gt):
    from rouge import Rouge

    try:
        return Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
    except Exception:  # LongBench scores an empty or unparseable prediction as 0
        return 0.0


def code_sim_score(pred, gt):
    from fuzzywuzzy import fuzz

    line = ""
    for candidate in pred.lstrip("\n").split("\n"):
        if "`" not in candidate and "#" not in candidate and "//" not in candidate:
            line = candidate
            break
    return fuzz.ratio(line, gt) / 100


LONGBENCH_METRIC = {"qasper": qa_f1_score, "multifieldqa_en": qa_f1_score, "hotpotqa": qa_f1_score,
                    "2wikimqa": qa_f1_score, "gov_report": rouge_score, "qmsum": rouge_score,
                    "lcc": code_sim_score, "repobench-p": code_sim_score}


def longbench_score(task, pred, answers):
    """Best score over the reference answers, x100 as in LongBench's tables."""
    return 100 * max(LONGBENCH_METRIC[task](pred, gt) for gt in answers)


# ------------------------------------------------------------------ RULER-style
ADJ = ("ancient bitter brave calm cheerful clever cold crisp curious damp eager empty fancy fierce "
       "gentle glossy grumpy hollow humble icy jolly lazy loud lucky mellow misty narrow noisy odd "
       "pale plain proud quiet rapid rough rusty salty shy silent sleepy smooth soft sour spicy "
       "steep sticky stormy sunny swift tame tidy tiny vast warm wild windy wise witty young").split()
NOUN = ("anchor apple arrow badge basket beacon blanket bottle bridge bucket cabin candle canyon "
        "castle cellar comet compass cradle crystal desert dolphin engine falcon feather fountain "
        "garden glacier hammer harbor helmet island jacket kettle ladder lantern meadow mirror "
        "orchard paddle pebble pillow planet pocket puzzle rocket saddle shovel spider stable "
        "statue thimble tunnel valley violin wagon whistle window").split()
VT_NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

NIAH_TEMPLATE = ("Some special magic {type_v} are hidden within the following text. Make sure to memorize it. "
                 "I will quiz you about the {type_v} afterwards.\n{context}\n"
                 "What are all the special magic {type_v} for {query} mentioned in the provided text?")
NIAH_PREFIX = " The special magic {type_v} for {query} mentioned in the provided text are"
NEEDLE = "One of the special magic {type_v} for {key} is: {value}."
VT_TEMPLATE = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
               "Question: Find all variables that are assigned the value {query} in the text above.")
VT_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables "
             "are assgined the value {query}, they are: ")
RULER_TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt")
RULER_MAXGEN = {"niah_single": 128, "niah_multikey": 128, "niah_multivalue": 128, "niah_multiquery": 128, "vt": 30}


def _sentences(text):
    return [s.strip() + "." for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


class RulerMaker:
    """Builds RULER-style examples of a target length (in tokens, chat template included)."""

    def __init__(self, tok, haystack_text):
        self.tok = tok
        self.sents = _sentences(haystack_text)
        self.sent_len = [len(tok(s, add_special_tokens=False).input_ids) for s in self.sents[:4000]]
        self.avg = sum(self.sent_len) / len(self.sent_len)

    def _ids(self, user, prefix):
        ids = self.tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                           tokenize=True, return_dict=False)
        return ids + self.tok(prefix, add_special_tokens=False).input_ids

    def _keys(self, rng, k):
        keys = set()
        while len(keys) < k:
            keys.add(f"{rng.choice(ADJ)}-{rng.choice(NOUN)}")
        return list(keys)

    def _value(self, rng):
        return str(rng.randint(1_000_000, 9_999_999))

    def _fit(self, build, length, avg=None):
        """Largest haystack (in units of `avg` tokens, default a sentence) whose prompt is at
        most `length` tokens."""
        lo, hi = 1, max(2, int(length / (avg or self.avg) * 1.5))
        best = build(lo)
        while lo <= hi:
            mid = (lo + hi) // 2
            ex = build(mid)
            if len(ex["ids"]) <= length:
                best, lo = ex, mid + 1
            else:
                hi = mid - 1
        return best

    def niah(self, task, length, seed):
        rng = random.Random(seed)
        n_keys = {"niah_single": 1, "niah_multikey": 4, "niah_multivalue": 1, "niah_multiquery": 4}[task]
        n_vals = 4 if task == "niah_multivalue" else 1
        keys = self._keys(rng, n_keys)
        needles = [(k, self._value(rng)) for k in keys for _ in range(n_vals)]
        queried = keys if task == "niah_multiquery" else keys[:1]  # multikey: the others are distractors
        answers = [v for k, v in needles if k in queried]
        type_v = "numbers"
        start = rng.randrange(0, len(self.sents) - 6000)
        depths = [rng.random() for _ in needles]
        order = list(range(len(needles)))
        rng.shuffle(order)
        query = ", ".join(queried)

        def build(n_sent):
            hay = self.sents[start:start + n_sent]
            pos = sorted((int(d * len(hay)), i) for d, i in zip(depths, order))
            parts, last = [], 0
            for p, i in pos:
                parts += hay[last:p] + [NEEDLE.format(type_v="number", key=needles[i][0], value=needles[i][1])]
                last = p
            parts += hay[last:]
            user = NIAH_TEMPLATE.format(type_v=type_v, context=" ".join(parts), query=query)
            return {"ids": self._ids(user, NIAH_PREFIX.format(type_v=type_v, query=query)),
                    "answers": answers, "depths": sorted(depths)}

        return self._fit(build, length)

    def vt(self, length, seed, hops=4):
        rng = random.Random(seed)
        names = {}  # a dict, not a set: set order varies with the hash seed, so runs differed
        while len(names) < hops + 1:
            names["".join(rng.choice(string.ascii_uppercase) for _ in range(5))] = None
        names = list(names)
        value = str(rng.randint(10000, 99999))
        chain = [f"VAR {names[0]} = {value}."] + [f"VAR {names[i + 1]} = VAR {names[i]}." for i in range(hops)]
        depths = sorted(rng.random() for _ in chain)

        def build(n_noise):
            noise = [VT_NOISE] * n_noise
            parts, last = [], 0
            for d, stmt in zip(depths, chain):
                p = int(d * n_noise)
                parts += noise[last:p] + [stmt + " "]
                last = p
            parts += noise[last:]
            user = VT_TEMPLATE.format(context="".join(parts), query=value)
            return {"ids": self._ids(user, VT_PREFIX.format(num_v=hops + 1, query=value)),
                    "answers": names, "depths": depths}

        return self._fit(build, length, avg=len(self.tok(VT_NOISE, add_special_tokens=False).input_ids))

    def make(self, task, length, seed):
        ex = self.vt(length, seed) if task == "vt" else self.niah(task, length, seed)
        ex["n_tokens"] = len(ex["ids"])
        return ex


def ruler_score(pred, answers):
    """RULER's string_match_all: share of the expected values that appear in the output, x100."""
    return 100 * sum(a.lower() in pred.lower() for a in answers) / len(answers)


# ------------------------------------------------------------------ MATH-500
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_math500(path="data/math500/test.jsonl", limit=None, seed=0, levels=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if levels:
        rows = [r for r in rows if r.get("level") in levels]
    if limit and limit < len(rows):  # a fixed random subset, stable across runs
        rows = random.Random(seed).sample(rows, limit)
    return rows


def math_ids(tok, row):
    return tok.apply_chat_template([{"role": "user", "content": f"{row['problem']}\n{MATH_INSTRUCTION}"}],
                                   add_generation_prompt=True, tokenize=True, return_dict=False)


def last_boxed(text):
    """Content of the last \\boxed{...} in text (braces balanced), or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    m = re.compile(r"\s*\{").match(text, i + len("\\boxed"))  # the brace must open the argument
    if m is None:
        return None
    j = m.end() - 1
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def math_correct(pred_text, answer):
    """Math-Verify equivalence of the last boxed answer with the reference."""
    from math_verify import parse, verify

    boxed = last_boxed(pred_text)
    if boxed is None:
        return False
    try:
        return bool(verify(parse(f"${answer}$"), parse(f"${boxed}$")))
    except Exception:
        return False
