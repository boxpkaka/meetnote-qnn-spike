# Server Dataset Preparation

MeetNote's server-side model experiments consume a model-independent canonical
dataset. Raw AliMeeting and MeetingBank files remain unchanged under
`/data/data`; generated JSONL is external to this repository.

## Build

```bash
python3 tools/prepare_meeting_datasets.py \
  --data-root /data/data \
  --output-dir /data/data/MeetNoteExperiments/v1 \
  --usage-scope internal-research
```

The command refuses to replace an existing output directory. A new version
must use a new directory such as `v2`, which prevents an experimental corpus
from changing silently beneath retained results.

## Contract

Each JSONL record uses schema `meetnote.experimental.meeting.v1` and retains the
source transcript without semantic cleanup.

- AliMeeting uses one far-field TextGrid per meeting as the canonical annotated
  transcript. Near- and far-field wav files are media variants on that same
  record, never separate samples. Official interval tiers provide speaker,
  timestamps, and contiguous `utt-0..utt-N` evidence IDs.
- MeetingBank preserves the official transcript and reference summary. Its
  `utterances` list is empty because the source has no reliable speaker/timing
  boundaries. Tokenizer-aware evidence chunks belong to a later fixture layer;
  this converter does not invent sentence boundaries.
- Official train/eval/test divisions map to train/validation/test. Calibration
  data must use train only.

`manifest.json` records source licenses and hashes, verifies every AliMeeting
archive against the official `SHA256SUMS`, records output sizes and hashes, and
captures the deterministic transformation contract without host-specific
absolute paths. `validation-report.json`
records the structural checks completed during generation. `stats.json`
contains only descriptive corpus statistics; counts and lengths are not
semantic quality gates.

MeetingBank is licensed CC BY-NC-SA 4.0. Keep it limited to internal research
unless the intended product use receives a separate license review. The tool
rejects `--usage-scope commercial-product`; a future commercial corpus must use
a separately reviewed source and contract rather than bypassing this gate.
