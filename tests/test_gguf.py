import os
from pathlib import Path
import struct
import tempfile
import unittest

from redlite.gguf import read_index, write_expert_manifest
from redlite.expert_cache import ExpertLRUCache
from redlite.streaming import streaming_plan


def _s(value: str) -> bytes:
    b = value.encode()
    return struct.pack('<Q', len(b)) + b


def make_gguf(path: str) -> None:
    meta = bytearray()
    meta += _s('general.alignment') + struct.pack('<I', 4) + struct.pack('<I', 32)
    meta += _s('general.architecture') + struct.pack('<I', 8) + _s('qwen3next')
    tensors = bytearray()
    specs = [
        ('blk.0.ffn_gate_up_exps.weight', (2, 2, 4), 0),
        ('blk.0.ffn_down_exps.weight', (2, 2, 4), 128),
        ('blk.0.attn_norm.weight', (4,), 256),
    ]
    for name, dims, off in specs:
        tensors += _s(name)
        tensors += struct.pack('<I', len(dims))
        for d in dims:
            tensors += struct.pack('<Q', d)
        tensors += struct.pack('<I', 0)
        tensors += struct.pack('<Q', off)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(specs), 2)
    prefix = header + meta + tensors
    data_start = (len(prefix) + 31) & ~31
    with open(path, 'wb') as f:
        f.write(prefix)
        f.write(b'\x00' * (data_start - len(prefix)))
        f.write(bytes(range(128)))
        f.write(bytes(range(128)))
        f.write(b'Z' * 128)


class GGUFTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.gguf')
        os.close(fd)
        make_gguf(self.path)

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def test_indexes_routed_experts(self):
        idx = read_index(self.path)
        self.assertEqual(idx.version, 3)
        self.assertEqual(len(idx.routed_tensors), 2)
        self.assertEqual(idx.routed_bytes, 256)
        records = idx.expert_records()
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].total_bytes, 64)
        self.assertEqual(len(records[0].segments), 2)

    def test_manifest(self):
        idx = read_index(self.path)
        out = Path(self.path + '.tsv')
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        write_expert_manifest(idx, out)
        lines = [x for x in out.read_text().splitlines() if not x.startswith('#')]
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith('0\t0\t64\t'))

    def test_reference_lru_reads_real_slices(self):
        records = read_index(self.path).expert_records()
        with ExpertLRUCache(self.path, 128) as cache:
            first = cache.get(records[0])
            self.assertEqual(len(first), 64)
            cache.get(records[0])
            cache.get(records[1])
            cache.get(records[2])
            self.assertGreaterEqual(cache.stats.hits, 1)
            self.assertGreaterEqual(cache.stats.evictions, 1)
            self.assertGreater(cache.stats.bytes_read, 0)

    def test_streaming_plan_separates_routed_payload(self):
        plan = streaming_plan(self.path, ram_gib=24, cache_gib=4, reserve_gib=4.5)
        self.assertEqual(plan.expert_records, 4)
        self.assertEqual(plan.layers, 1)
        self.assertGreater(plan.routed_expert_gib, 0)
        self.assertTrue(plan.safe)


if __name__ == '__main__':
    unittest.main()
