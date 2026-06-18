from collections import Counter, defaultdict
import sys

ref_path = "egs2/uaspeech/asr2/exp/trim/asr_train_discrete_asr_e_branchformer1_1gpu_raw_wavlm_large_21_km2000_bpe_rm4000_trim_all_mic_char_ts_sp/decode_ctc0.3_asr_model_valid.acc.ave/org/trim/dev/score_wer/ref.trn"
hyp_path = ref_path.replace('"'"'ref.trn'"'"','"'"'hyp.trn'"'"')

def load(path):
    data = []
    with open(path, encoding='"'"'utf-8'"'"') as f:
        for line in f:
            line = line.rstrip('"'"'\n'"'"')
            if not line:
                continue
            if '"'"'\t'"'"' in line:
                text, utt = line.split('"'"'\t'"'"', 1)
            else:
                text, utt = line.split('"'"'('"'"', 1)
                utt = '"'"'('"'"' + utt
            data.append((text.strip(), utt.strip()))
    return data

ref = load(ref_path)
hyp = load(hyp_path)
assert len(ref) == len(hyp)
sub = Counter()
del_c = Counter()
ins_c = Counter()
total_by_text = Counter()
err_by_text = Counter()

def align(r, h):
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = '"'"'del'"'"'
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = '"'"'ins'"'"'
    for i in range(1, n+1):
        ri = r[i-1]
        for j in range(1, m+1):
            cost = 0 if ri == h[j-1] else 1
            vals = [
                (dp[i-1][j] + 1, '"'"'del'"'"'),
                (dp[i][j-1] + 1, '"'"'ins'"'"'),
                (dp[i-1][j-1] + cost, '"'"'sub'"'"'),
            ]
            dp[i][j], bt[i][j] = min(vals, key=lambda x: x[0])
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == '"'"'sub'"'"':
            ops.append(('"'"'sub'"'"', r[i-1], h[j-1]))
            i -= 1
            j -= 1
        elif op == '"'"'del'"'"':
            ops.append(('"'"'del'"'"', r[i-1], None))
            i -= 1
        elif op == '"'"'ins'"'"':
            ops.append(('"'"'ins'"'"', None, h[j-1]))
            j -= 1
        else:
            break
    ops.reverse()
    return ops

for (rtext, _), (htext, _) in zip(ref, hyp):
    r_tokens = rtext.split()
    h_tokens = htext.split()
    total_by_text[rtext] += 1
    if r_tokens != h_tokens:
        err_by_text[rtext] += 1
    ops = align(r_tokens, h_tokens)
    for op, rtok, htok in ops:
        if op == '"'"'sub'"'"' and rtok != htok:
            sub[(rtok, htok)] += 1
        elif op == '"'"'del'"'"':
            del_c[rtok] += 1
        elif op == '"'"'ins'"'"':
            ins_c[htok] += 1

print("Top substitutions (dev):")
for (r,h), c in sub.most_common(10):
    print(f"{c:3d}  {r} -> {h}")

print("\nHardest prompts (>=10 occurrences):")
items = []
for text, tot in total_by_text.items():
    if tot >= 10:
        err = err_by_text[text]
        items.append((err / tot, err, tot, text))
for rate, err, tot, text in sorted(items, key=lambda x: (-x[0], -x[1]))[:10]:
    print(f"{rate*100:4.1f}% ({err}/{tot})  {text}")

print("\nEasiest prompts (>=10 occurrences):")
for rate, err, tot, text in sorted(items, key=lambda x: (x[0], -x[2]))[:10]:
    print(f"{rate*100:4.1f}% ({err}/{tot})  {text}")
# PY'
# Top substitutions (dev):
#  27  THERE -> THEIR
#  19  FOR -> FOUR
#  16  TO -> TWO
#  11  RIGHT -> WRITE
#   7  THEN -> THAN
#   6  MASSACHUSETTS -> MASSACHUSET
#   5  AND -> AN
#   5  FOIL -> OIL
#   5  DOWNWARD -> DOWNWAR
#   4  FORGETFULNESS -> FORGETFULNES

# Hardest prompts (>=10 occurrences):
# 100.0% (32/32)  THERE
# 87.5% (21/24)  FOR
# 78.3% (18/23)  TO
# 68.8% (11/16)  RIGHT
# 52.6% (10/19)  THEN
# 50.0% (6/12)  MASSACHUSETTS
# 47.1% (8/17)  ARE
# 41.7% (5/12)  PROFESSIONALS
# 40.0% (6/15)  FOIL
# 40.0% (4/10)  FORGETFULNESS

# Easiest prompts (>=10 occurrences):
#  0.0% (0/35)  X-RAY
#  0.0% (0/34)  EIGHT
#  0.0% (0/34)  PAPA
#  0.0% (0/32)  ECHO
#  0.0% (0/32)  VICTOR
#  0.0% (0/32)  WHISKEY
#  0.0% (0/32)  CONTROL
#  0.0% (0/31)  JULIET
#  0.0% (0/30)  ENTER
#  0.0% (0/29)  PEOPLE


