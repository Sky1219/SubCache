"""TinyStories validation stories and the recall probe.

Stories come from roneneldan/TinyStories' TinyStories-valid.txt (stories separated by
<|endoftext|>), each tokenized with the GPT-2 tokenizer and prefixed with <|endoftext|> (id 50256),
the separator the model was trained with, so the first word is predicted too and position 0 is a
fixed attention sink.

Recall probe. A target token at position j is a *recall token* if
  * j is in the second half of the story (j >= len / 2),
  * it starts a word (GPT-2 marks that with a leading space) and is alphabetic,
  * it is a name (capitalized, not a sentence-opening function word) or a content word
    (lowercase, >= 4 letters, not a stopword),
  * the same word occurred before, and its most recent earlier occurrence is at least `gap`
    tokens back (j - i >= gap). A word that also appeared recently does not count: a recency
    window would trivially hold it.
Matching is on the word string, so the story-opening "Lily" (no leading space, a different
token) matches a later " Lily". Only the word-initial token is scored; the rest of a multi-token
word is not.
"""

import random

from huggingface_hub import hf_hub_download

BOS = 50256

STOPWORDS = set("""
a about above after again against all also am an and any are as at away back be because been before
being below between both but by came can come could did do does doing done down during each even
ever every few for from get gets got had has have having he her here hers herself him himself his
how i if in into is it its itself just know knew let like little made make many may me more most
much must my myself never new no nor not now of off oh on once one only or other our ours out over
own put said same saw say says see she should so some something soon still such take than that
the their theirs them then there these they thing things this those through to too took up upon
us very want wanted was way we went were what when where which while who whom why will with would
yes you your yours very really always went goes going gone got there their very time day
""".split())

# Capitalized words that open sentences or dialogue rather than name a character or thing.
OPENERS = set("""
The A An And But So Then When One Once They He She It I We You His Her Their There This That
These Those What Why How Who Where Yes No Oh Can Do Did Let Please Thank Thanks Look Come Go After
Every Some All If In On At As Just Now Soon Later Finally Suddenly Today Mom Dad Mommy Daddy
""".split())


def load_stories(tok, n=500, seed=0, min_len=160, max_len=512, exclude=()):
    """`n` stories drawn uniformly (random.Random(seed)) from the validation stories whose length,
    including the leading <|endoftext|>, is in [min_len, max_len] and whose index is not in
    `exclude`. Returns dicts with the story's index in the file and its token ids."""
    path = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        texts = [s.strip() for s in f.read().split("<|endoftext|>")]
    eligible = []
    for i, text in enumerate(texts):
        if not text:
            continue
        ids = [BOS] + tok(text)["input_ids"]
        if min_len <= len(ids) <= max_len and i not in exclude:
            eligible.append({"story": i, "ids": ids})
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} stories in [{min_len}, {max_len}] tokens")
    return random.Random(seed).sample(eligible, n)


def word_kind(piece):
    """'name', 'word' or None for one decoded token."""
    starts_word = piece.startswith(" ")
    w = piece.strip()
    if not w.isalpha() or not w.isascii():
        return None, w
    if w[0].isupper():
        return ("name" if w not in OPENERS and len(w) >= 2 else None), w
    if starts_word and len(w) >= 4 and w.lower() not in STOPWORDS:
        return "word", w
    return None, w


def recall_targets(ids, tok, gap=32):
    """{position j: (kind, distance to the previous occurrence)} for the recall tokens of a story
    (see the module docstring). Position j means token ids[j], predicted from the logits at j - 1."""
    pieces = [tok.decode([i]) for i in ids]
    last_seen = {}
    out = {}
    half = len(ids) / 2
    for j, piece in enumerate(pieces):
        if j == 0:
            continue
        kind, w = word_kind(piece)
        # A word occurrence counts as seen whether or not it starts with a space ("Lily" at the
        # story start), but only word-initial pieces can be targets.
        word_start = piece.startswith(" ") or j == 1 or pieces[j - 1].endswith(("\n", '"', "(", "'"))
        if not word_start or not w.isalpha():
            continue
        if kind is not None and piece.startswith(" ") and j >= half and w in last_seen:
            dist = j - last_seen[w]
            if dist >= gap:
                out[j] = (kind, dist)
        last_seen[w] = j
    return out
