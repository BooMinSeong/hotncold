from datasets import load_dataset

ds = "ENSEONG/full-math-private-n256-Llama-3.2-3B-Instruct-bon"
T, S = "0.7", "0"
final_rev = f"ENSEONG_math-private--T-{T}--top_p-1.0--n-256--seed-{S}--agg_strategy-last--chunk-0_5000"
gen_rev = final_rev + "-gen"

print("=== loading FINAL ===")
d_final = load_dataset(ds, revision=final_rev, split="train")
print(f"rows: {len(d_final)}")
print(f"columns ({len(d_final.column_names)}): {d_final.column_names}")

print()
print("=== loading -gen ===")
d_gen = load_dataset(ds, revision=gen_rev, split="train")
print(f"rows: {len(d_gen)}")
print(f"columns ({len(d_gen.column_names)}): {d_gen.column_names}")

print()
fc = set(d_final.column_names)
gc = set(d_gen.column_names)
print(f"only in FINAL: {sorted(fc - gc)}")
print(f"only in -gen:  {sorted(gc - fc)}")
fp = d_final[0]["problem"]
gp = d_gen[0]["problem"]
print(f"final[0] problem == gen[0] problem: {fp == gp}")
