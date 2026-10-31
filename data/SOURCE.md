# Benchmark data provenance

The benchmark uses the complete official GSM8K main dataset: 7,473 official training rows
and 1,319 official test rows. No prefix-only pool is used.

Source repository: https://github.com/openai/grade-school-math

The experiment files were downloaded through jsDelivr's GitHub mirror using `curl --compressed`,
because direct GitHub downloads were interrupted in this environment. TLS certificate checking
remained enabled. Complete byte counts and all JSONL rows were validated before use.

The test file's Git blob SHA1 was independently obtained from the GitHub contents API and
verified. SHA256 and Git blob SHA1 checksums for both files are recorded below and in
`raw/gsm8k_manifest.json`; the training hash is a reproducibility checksum.

Retrieved: 2026-10-31T08:33:40.214485+00:00

## train

- Original: https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/train.jsonl
- Download: https://cdn.jsdelivr.net/gh/openai/grade-school-math@master/grade_school_math/data/train.jsonl
- Rows: 7473; bytes: 4166206
- SHA256: `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`
- Git blob SHA1: `7d97154b91aef28d01c3301741c81ce90039a4b1`

## test

- Original: https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl
- Download: https://cdn.jsdelivr.net/gh/openai/grade-school-math@master/grade_school_math/data/test.jsonl
- Rows: 1319; bytes: 749738
- SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- Git blob SHA1: `e4c2ff4942b9a78bd74f04141224c11e28d12dc9`

Training and validation are sampled only from official train. Final evaluation is sampled
only from official test, with question-hash deduplication across splits.

Reproduce from the repository root (no network request if complete valid files exist):

```sh
python scripts/download_data.py
```
