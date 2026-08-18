import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_meeting_datasets",
    ROOT / "tools/prepare_meeting_datasets.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareMeetingDatasetsTest(unittest.TestCase):
    TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "ROOM_MEETING_F_SPK0001"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0.1
            xmax = 0.9
            text = "hello"
'''

    def test_parse_textgrid_and_make_stable_evidence_ids(self):
        content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 3.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "R0001_M0001_F_SPK0002"
        xmin = 0
        xmax = 3.5
        intervals: size = 1
        intervals [1]:
            xmin = 1.25
            xmax = 2.0
            text = "second"
    item [2]:
        class = "IntervalTier"
        name = "R0001_M0001_M_SPK0001"
        xmin = 0
        xmax = 3.5
        intervals: size = 2
        intervals [1]:
            xmin = 0.1
            xmax = 0.9
            text = "first"
        intervals [2]:
            xmin = 2.2
            xmax = 2.8
            text = "quoted ""value"""
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.TextGrid"
            path.write_text(content, encoding="utf-8")
            duration_ms, raw = MODULE.parse_textgrid(path)

        self.assertEqual(3500, duration_ms)
        utterances = MODULE.canonicalize_intervals(raw)
        self.assertEqual(["utt-0", "utt-1", "utt-2"], [item["id"] for item in utterances])
        self.assertEqual(["S1", "S2", "S1"], [item["speaker"] for item in utterances])
        self.assertEqual("quoted \"value\"", utterances[2]["text"])

    def test_milliseconds_use_decimal_half_up_rounding(self):
        self.assertEqual(1235, MODULE.seconds_to_ms(Decimal("1.2345")))

    def test_combined_dataset_rejects_commercial_usage(self):
        with self.assertRaisesRegex(ValueError, "CC BY-NC-SA"):
            MODULE.validate_usage_scope("commercial-product")

    def make_dataset_source(self, root: Path) -> None:
        archive = root / "AliMeeting/archives/source.tar.gz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"archive")
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksums = root / "AliMeeting/checksums/SHA256SUMS"
        checksums.parent.mkdir(parents=True)
        checksums.write_text(f"{checksum}  source.tar.gz\n", encoding="utf-8")

        for index, split in enumerate(("train", "validation", "test"), start=1):
            layout = MODULE.ALI_SPLITS[split]
            meeting_id = f"meeting-{index}"
            textgrids = root / layout["far_textgrids"]
            far_audio = root / layout["far_audio"]
            near_audio = root / layout["near_audio"]
            for directory in (textgrids, far_audio, near_audio):
                directory.mkdir(parents=True)
            (textgrids / f"{meeting_id}.TextGrid").write_text(
                self.TEXTGRID, encoding="utf-8"
            )
            (far_audio / f"{meeting_id}_far.wav").write_bytes(b"far")
            (near_audio / f"{meeting_id}_near.wav").write_bytes(b"near")

            meetingbank = root / MODULE.MEETINGBANK_SPLITS[split]
            meetingbank.parent.mkdir(parents=True, exist_ok=True)
            meetingbank.write_text(
                json.dumps(
                    {
                        "uid": f"bank-{index}",
                        "id": index,
                        "transcript": "meeting transcript",
                        "summary": "meeting summary",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    def test_manifest_is_path_independent_and_archives_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "source-a"
            source_b = root / "source-b"
            self.make_dataset_source(source_a)
            self.make_dataset_source(source_b)
            manifest_a = MODULE.build_dataset(source_a, root / "output-a")
            manifest_b = MODULE.build_dataset(source_b, root / "output-b")
            self.assertEqual(manifest_a, manifest_b)
            self.assertNotIn("data_root", manifest_a)

            archive = source_b / "AliMeeting/archives/source.tar.gz"
            archive.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.parse_archive_checksums(source_b)


if __name__ == "__main__":
    unittest.main()
