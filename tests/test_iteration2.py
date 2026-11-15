"""Correctness checks for inference-time patch selection. Synthetic tensors are not benchmark scores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reidvlm.encoders.clip_reid import CLIPReIDViT, ResidualAttentionBlock, VisionTransformer
from reidvlm.evaluation.metrics import evaluate_rankings
from reidvlm.persistence import RunStore
from reidvlm.pruning.selection import (
    SelectionConfig,
    cls_patch_scores,
    content_id_from_tensor,
    gather_kept,
    random_patch_indices,
    reference_patch_scores,
    same_selection,
)
from reidvlm.pruning.study import paired_query_changes, resume_embed


def _tiny() -> VisionTransformer:
    model = VisionTransformer(32, 32, 16, 16)
    model.eval()
    return model


class SelectionTests(unittest.TestCase):
    def test_full_token_path_matches_forced_selection(self):
        torch.manual_seed(0)
        model = _tiny()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            _a, full, _proj = model(images, keep_ratio=1.0)
            forced = SelectionConfig("cls_attention", 6, 4, None, "reference", 4)
            _b, selected, _proj2 = model(images, selection=forced)
        difference = (full - selected).abs().max().item()
        self.assertLess(difference, 1e-5, f"max abs descriptor difference {difference}")
        self.assertEqual(model.last_tokens_out, 5)
        self.assertEqual(model.last_tokens_per_block[0], 5)
        self.assertEqual(model.last_tokens_per_block[-1], 5)

    def test_short_sequence_shape_order_and_cls(self):
        torch.manual_seed(1)
        model = _tiny()
        images = torch.randn(2, 3, 32, 32)
        config = SelectionConfig("cls_attention", 3, 2, None, "reference", 4)
        with torch.inference_mode():
            model(images, selection=config)
        index = model.last_kept_patch_index
        self.assertEqual(tuple(index.shape), (2, 2))
        self.assertTrue(torch.all(index[:, 0] < index[:, 1]))
        self.assertEqual(model.last_tokens_per_block[2], 5)
        self.assertEqual(model.last_tokens_per_block[3], 3)
        self.assertEqual(len(model.last_tokens_per_block), 12)

    def test_positional_embedding_is_one_parameter(self):
        model = _tiny()
        self.assertEqual(model.positional_embedding.ndim, 2)
        self.assertEqual(model.positional_embedding.shape[0], 5)

    def test_random_reproducible_across_order_and_rejects_duplicates(self):
        first = [content_id_from_tensor(torch.arange(i, i + 12).float().view(3, 2, 2)) for i in range(6)]
        reversed_ids = list(reversed(first))
        masks = [random_patch_indices(11, item, 8, 3) for item in first]
        again = [random_patch_indices(11, item, 8, 3) for item in reversed_ids]
        self.assertEqual(masks, list(reversed(again)))
        for mask in masks:
            self.assertEqual(len(mask), len(set(mask)))
            self.assertEqual(mask, sorted(mask))
        other = [random_patch_indices(23, item, 8, 3) for item in first]
        self.assertNotEqual(masks, other)

    def test_reference_and_cls_row_scores(self):
        torch.manual_seed(2)
        block = ResidualAttentionBlock(64, 4)
        block.eval()
        tokens = torch.randn(6, 3, 64)
        with torch.inference_mode():
            _out, reference = block(tokens, return_attn=True, attention_backend="reference")
            normed = block.ln_1(tokens)
            direct = cls_patch_scores(block.attn, normed)
            weights = block.attn(normed, normed, normed, need_weights=True, average_attn_weights=True)[1]
            from_weights = reference_patch_scores(weights)
        self.assertEqual(tuple(reference.shape), (3, 5))
        gap = (reference - direct).abs().max().item()
        weight_gap = (reference - from_weights).abs().max().item()
        self.assertLess(weight_gap, 1e-6, f"reference scores differ from need_weights by {weight_gap}")
        selected_gap = (reference.argsort(dim=1) - direct.argsort(dim=1)).abs().max().item()
        print(f"cls_row_max_abs_score_gap={gap:.3e} argsort_index_gap={selected_gap}")
        # Equivalence is required before cls_row may replace reference. Do not loosen this.
        self.assertLess(gap, 1e-4, f"cls_row is not equivalent to reference; max abs gap {gap:.3e}")

    def test_gather_preserves_cls_and_spatial_order(self):
        tokens = torch.arange(10, dtype=torch.float32).view(5, 2, 1)
        index = torch.tensor([[2, 0], [3, 1]])
        kept = gather_kept(tokens, index)
        self.assertTrue(torch.equal(kept[0], tokens[0]))
        self.assertTrue(torch.equal(kept[1, 0], tokens[3, 0]))
        self.assertTrue(torch.equal(kept[2, 0], tokens[1, 0]))
        self.assertEqual(tuple(kept.shape), (3, 2, 1))

    def test_batchnorm_statistics_unchanged(self):
        model = CLIPReIDViT(32, 32, 16, False, 6, 1.0)
        model.eval()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            model(torch.randn(2, 3, 32, 32))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def test_parameters_and_buffers_unchanged(self):
        model = _tiny()
        model.eval()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            model(torch.randn(1, 3, 32, 32), selection=SelectionConfig("random", 4, 2, 11, "reference", 4), content_ids=["abc"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            SelectionConfig("cls_attention", 12, 4, None, "reference", 4).validate()
        with self.assertRaises(ValueError):
            SelectionConfig("random", 6, 4, None, "reference", 4).validate()
        with self.assertRaises(ValueError):
            SelectionConfig("none", 6, 3, None, "reference", 4).validate()
        with self.assertRaises(ValueError):
            SelectionConfig("cls_attention", 6, 0, None, "reference", 4).validate()
        with self.assertRaises(ValueError):
            same_selection(
                SelectionConfig("cls_attention", 6, 2, None, "reference", 4),
                SelectionConfig("cls_attention", 9, 2, None, "reference", 4),
            )

    def test_resume_matches_uninterrupted(self):
        identity = {"selection_version": "selection-v1", "strategy": "random", "seed": 11, "rows": 4}
        ids = [f"img-{index}" for index in range(4)]

        def embed(missing: list[str]) -> np.ndarray:
            return np.array([[float(int(item.split("-")[1])), 1.0] for item in missing], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            full = RunStore(Path(directory) / "full")
            full.prepare()
            expected = resume_embed(full, identity, ids, embed)
            partial = RunStore(Path(directory) / "partial")
            partial.prepare()
            resume_embed(partial, identity, ids[:2], embed)
            resumed = resume_embed(partial, identity, ids, embed)
        self.assertTrue(np.allclose(expected, resumed))
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            store.prepare()
            store.save_feature_table(identity, {"img-0": np.array([1.0, 1.0], dtype=np.float32)})
            other = dict(identity)
            other["seed"] = 23
            self.assertIsNone(store.load_feature_table(other))
            key = store.feature_key(identity)
            store.publish_json(
                f"features/{key}.json",
                {"identity": other, "count": 1},
                replace=True,
            )
            with self.assertRaises(ValueError):
                store.load_feature_table(identity)


class MetricFixture(unittest.TestCase):
    def test_protocol_filters(self):
        # query pid 1 camera 0. Gallery: same-camera hit, junk, same-camera other person, two cross-camera hits.
        orders = np.array([[0, 1, 2, 3, 4]])
        q_pids = np.array([1])
        q_cams = np.array([0])
        g_pids = np.array([1, -1, 2, 1, 1])
        g_cams = np.array([0, 0, 0, 1, 2])
        result = evaluate_rankings(orders, q_pids, g_pids, q_cams, g_cams, k=5)
        self.assertEqual(result.valid_queries, 1)
        self.assertEqual(result.excluded_queries, 0)
        self.assertFalse(result.per_query[0].top1_correct)
        self.assertAlmostEqual(result.per_query[0].ap, (0.5 + 2 / 3) / 2)
        self.assertEqual(result.per_query[0].shortlist_positive_count, 2)
        self.assertEqual(result.rank1, 0.0)
        self.assertEqual(result.rank5, 1.0)
        mixed = evaluate_rankings(
            np.array([[0, 1], [0, 1]]),
            np.array([1, 3]),
            np.array([1, 1]),
            np.array([0, 0]),
            np.array([0, 1]),
            k=5,
        )
        # Query pid 3 has two kept gallery rows and neither is a positive, so it is excluded.
        self.assertEqual(mixed.valid_queries, 1)
        self.assertEqual(mixed.excluded_queries, 1)
        self.assertTrue(mixed.per_query[0].top1_correct)
        self.assertTrue(mixed.per_query[1].excluded)
        with self.assertRaises(ValueError):
            evaluate_rankings(np.array([[0]]), np.array([1]), np.array([1]), np.array([0]), np.array([0]))

    def test_paired_changes(self):
        changes = paired_query_changes([0.2, 0.4, 0.5], [0.3, 0.1, 0.5])
        self.assertEqual(changes, {"improved": 1, "worsened": 1, "unchanged": 1, "queries": 3})


if __name__ == "__main__":
    unittest.main()
