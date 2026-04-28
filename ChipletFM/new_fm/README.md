# ChipletFM New Flow Matching

This directory is a clean Flow Matching rewrite for chiplet/layout generation.

It is intentionally independent from the legacy model, training, sampling,
guidance, and legalizer code. Legacy project files may only be used as dataset
and file-format inputs.

Initial scope:

- load existing graph placement datasets into a new schema
- train a small Flow Matching baseline
- sample placements from checkpoints
- evaluate HPWL and simple geometric legality

Coordinate convention: model/schema positions are node center coordinates.
Legacy placement files are read as lower-left coordinates and converted in the
dataset adapter.

Runtime outputs:

- training logs and checkpoints are written under `new_fm/log/`
- sampling and evaluation outputs should also be written under `new_fm/log/`
- `eval_num` controls how many test samples are evaluated
- `out_num` controls how many visualization PNGs are emitted

Example commands:

```powershell
python -m new_fm.train.train --config new_fm/configs/smoke.yaml

python -m new_fm.train.train --config new_fm/configs/raw_v2.yaml

python -m new_fm.sampling.sample `
  --ckpt new_fm/log/smoke/best.ckpt `
  --split test `
  --output new_fm/log/smoke/samples `
  --eval-num 3 `
  --out-num 2 `
  --steps 5

python -m new_fm.sampling.eval `
  --ckpt new_fm/log/smoke/best.ckpt `
  --split test `
  --output new_fm/log/smoke/generated_eval/metrics.json `
  --eval-num 3 `
  --out-num 2
```
