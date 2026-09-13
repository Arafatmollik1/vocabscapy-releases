"""Batch-generate word illustrations with FLUX.2 Klein 4B. Run on a CUDA box: python generate.py prompts.csv out/"""
import csv, os, sys, time, hashlib
import torch
from diffusers import Flux2KleinPipeline

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16).to("cuda")
rows = list(csv.DictReader(open(src)))
t0 = time.time()
for i, r in enumerate(rows, 1):
    name = r["key"].replace("|", "_").replace("/", "_")
    path = os.path.join(out, f"{name}.png")
    if os.path.exists(path):
        continue
    seed = int(hashlib.md5(r["key"].encode()).hexdigest()[:8], 16)
    img = pipe(prompt=r["prompt"], height=768, width=768, guidance_scale=1.0, num_inference_steps=4,
               generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
    img.save(path)
    if i % 20 == 0:
        print(f"{i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
print("done", len(rows), f"{time.time()-t0:.0f}s")
