"""CPU Pallas interpret-mode and geometry tests for `qk_norm_rope_kernel`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from torchtitan.experiments.tpu.kernels import qk_norm_rope_kernel as kernel


class QkNormRopeKernelTest(parameterized.TestCase):

  def test_geometry_validation_accepts_and_rejects_expected_cases(self):
    self.assertIsNone(
        kernel.validate_geometry(
            batch=2,
            seq=2048,
            hidden_dim=18 * 64,
            heads=18,
            head_dim=64,
            cos_shape=(1, 2048, 64),
            sin_shape=(1, 2048, 64),
            block_s=512,
        )
    )
    self.assertIsNone(
        kernel.validate_geometry(
            batch=2,
            seq=32,
            hidden_dim=2 * 128,
            heads=2,
            head_dim=128,
            cos_shape=(2, 32, 128),
            sin_shape=(2, 32, 128),
            block_s=16,
        )
    )
    # Sub-64 head_dim (e.g. 32) is rejected.
    self.assertIn(
        "head_dim",
        kernel.validate_geometry(
            batch=1,
            seq=16,
            hidden_dim=4 * 32,
            heads=4,
            head_dim=32,
            cos_shape=(1, 16, 32),
            sin_shape=(1, 16, 32),
            block_s=16,
        )
        or "",
    )
    # Odd head count at D=64 violates 128-lane group alignment.
    self.assertIn(
        "group width",
        kernel.validate_geometry(
            batch=1,
            seq=16,
            hidden_dim=3 * 64,
            heads=3,
            head_dim=64,
            cos_shape=(1, 16, 64),
            sin_shape=(1, 16, 64),
            block_s=16,
        )
        or "",
    )
    # Projection width mismatch.
    self.assertIn(
        "projection width",
        kernel.validate_geometry(
            batch=1,
            seq=16,
            hidden_dim=256,
            heads=2,
            head_dim=64,
            cos_shape=(1, 16, 64),
            sin_shape=(1, 16, 64),
            block_s=16,
        )
        or "",
    )
    # Invalid cos/sin batch broadcast dimension (2 != 1 and 2 != batch=4).
    self.assertIn(
        "cos/sin shape",
        kernel.validate_geometry(
            batch=4,
            seq=16,
            hidden_dim=2 * 64,
            heads=2,
            head_dim=64,
            cos_shape=(2, 16, 64),
            sin_shape=(2, 16, 64),
            block_s=16,
        )
        or "",
    )
    # Non-8-sublane-aligned block_s.
    self.assertIn(
        "aligned tiles",
        kernel.validate_geometry(
            batch=1,
            seq=24,
            hidden_dim=2 * 64,
            heads=2,
            head_dim=64,
            cos_shape=(1, 24, 64),
            sin_shape=(1, 24, 64),
            block_s=12,
        )
        or "",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="d64_multi_tile_batched_qscale",
          batch=2,
          seq=32,
          block_s=16,
          heads=4,
          head_dim=64,
          cos_batch=2,
          q_scale=0.125,
          asymmetric_tables=False,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="d64_four_tiles_unbatched_asymmetric_tables",
          batch=2,
          seq=64,
          block_s=16,
          heads=4,
          head_dim=64,
          cos_batch=1,
          q_scale=1.0,
          asymmetric_tables=True,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="d128_multi_tile_bf16",
          batch=2,
          seq=32,
          block_s=16,
          heads=2,
          head_dim=128,
          cos_batch=1,
          q_scale=0.125,
          asymmetric_tables=True,
          dtype=jnp.bfloat16,
      ),
  )
  def test_interpret_forward_and_backward_match_reference(
      self,
      batch: int,
      seq: int,
      block_s: int,
      heads: int,
      head_dim: int,
      cos_batch: int,
      q_scale: float,
      asymmetric_tables: bool,
      dtype: jnp.dtype,
  ):
    hidden_dim = heads * head_dim
    eps = 1e-5
    keys = jax.random.split(jax.random.PRNGKey(42), 10)
    xq = (jax.random.normal(keys[0], (batch, seq, hidden_dim)) * 1.5 + 0.2).astype(dtype)
    xk = (jax.random.normal(keys[1], (batch, seq, hidden_dim)) * 1.2 - 0.1).astype(dtype)
    xv = jax.random.normal(keys[2], (batch, seq, hidden_dim)).astype(dtype)
    wq = (1.0 + 0.1 * jax.random.normal(keys[3], (hidden_dim,))).astype(dtype)
    wk = (1.0 + 0.1 * jax.random.normal(keys[4], (hidden_dim,))).astype(dtype)

    if asymmetric_tables:
      cos = jax.random.uniform(
          keys[5], (cos_batch, seq, head_dim), minval=-1.0, maxval=1.0
      ).astype(dtype)
      sin = jax.random.uniform(
          keys[6], (cos_batch, seq, head_dim), minval=-1.0, maxval=1.0
      ).astype(dtype)
    else:
      angles = jax.random.uniform(
          keys[5], (cos_batch, seq, head_dim // 2), minval=-3.0, maxval=3.0
      )
      emb = jnp.concatenate([angles, angles], axis=-1)
      cos = jnp.cos(emb).astype(dtype)
      sin = jnp.sin(emb).astype(dtype)

    gq = jax.random.normal(keys[7], (batch, heads, seq, head_dim)).astype(dtype)
    gk = jax.random.normal(keys[8], (batch, heads, seq, head_dim)).astype(dtype)
    gv = jax.random.normal(keys[9], (batch, heads, seq, head_dim)).astype(dtype)

    q_got, k_got, v_got = kernel.qk_norm_rope_fwd(
        xq,
        xk,
        xv,
        wq,
        wk,
        cos,
        sin,
        heads=heads,
        eps=eps,
        q_scale=q_scale,
        block_s=block_s,
        interpret=True,
    )
    dxq_got, dxk_got, dxv_got, dwq_got, dwk_got = kernel.qk_norm_rope_bwd(
        xq,
        xk,
        wq,
        wk,
        cos,
        sin,
        gq,
        gk,
        gv,
        eps=eps,
        q_scale=q_scale,
        block_s=block_s,
        interpret=True,
    )

    def ref_fn(xq_, xk_, xv_, wq_, wk_):
      return kernel.qk_norm_rope_reference(
          xq_.astype(jnp.float32),
          xk_.astype(jnp.float32),
          xv_.astype(jnp.float32),
          wq_.astype(jnp.float32),
          wk_.astype(jnp.float32),
          cos.astype(jnp.float32),
          sin.astype(jnp.float32),
          heads=heads,
          eps=eps,
          q_scale=q_scale,
      )

    (q_ref, k_ref, v_ref), vjp_fn = jax.vjp(ref_fn, xq, xk, xv, wq, wk)
    dxq_ref, dxk_ref, dxv_ref, dwq_ref, dwk_ref = vjp_fn(
        (gq.astype(jnp.float32), gk.astype(jnp.float32), gv.astype(jnp.float32))
    )

    tol = 1.5e-2 if dtype == jnp.bfloat16 else 1e-5
    for name, got, ref in (
        ("q", q_got, q_ref),
        ("k", k_got, k_ref),
        ("v", v_got, v_ref),
        ("dxq", dxq_got, dxq_ref),
        ("dxk", dxk_got, dxk_ref),
        ("dxv", dxv_got, dxv_ref),
        ("dwq", dwq_got, dwq_ref),
        ("dwk", dwk_got, dwk_ref),
    ):
      got_f = np.asarray(got, dtype=np.float32)
      ref_f = np.asarray(ref, dtype=np.float32)
      rel_err = np.linalg.norm(got_f - ref_f) / max(
          np.linalg.norm(ref_f), 1e-12
      )
      self.assertLessEqual(rel_err, tol, msg=f"Mismatch in {name}: {rel_err}")

  def test_folded_segment_lanes_match_plain_outputs(self):
    """Widened q/k carry the plain heads plus an exact document mask."""
    batch, seq, heads, head_dim, block_s = 2, 32, 4, 64, 16
    hidden_dim = heads * head_dim
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    xq, xk, xv = (
        jax.random.normal(keys[i], (batch, seq, hidden_dim)).astype(jnp.bfloat16)
        for i in range(3)
    )
    wq = (1.0 + 0.1 * jax.random.normal(keys[3], (hidden_dim,))).astype(jnp.bfloat16)
    wk = (1.0 + 0.1 * jax.random.normal(keys[4], (hidden_dim,))).astype(jnp.bfloat16)
    angles = jax.random.uniform(keys[5], (1, seq, head_dim // 2), minval=-3.0, maxval=3.0)
    emb = jnp.concatenate([angles, angles], axis=-1)
    cos, sin = jnp.cos(emb).astype(jnp.bfloat16), jnp.sin(emb).astype(jnp.bfloat16)
    # Row 0: documents of 10 / 12 / 10 tokens; row 1: one document + padding.
    rel = jnp.array(
        [[0] * 10 + [1] * 12 + [2] * 10, [0] * 20 + [1] * 12], jnp.int32
    )[..., None]
    common = dict(heads=heads, eps=1e-5, q_scale=0.125, block_s=block_s,
                  interpret=True)

    q, k, v = kernel.qk_norm_rope_fwd(xq, xk, xv, wq, wk, cos, sin, **common)
    qa, ka, va = kernel.qk_norm_rope_fwd(
        xq, xk, xv, wq, wk, cos, sin, segment_rel=rel, **common
    )
    self.assertEqual(qa.shape, (batch, heads, seq, 128))

    # Odd heads hold their values in the upper lane half; swapping the halves
    # of odd heads gives every head the [values | segment code] layout.
    def canonical(t):
      return t.at[:, 1::2].set(jnp.roll(t[:, 1::2], head_dim, axis=-1))

    qa, ka = canonical(qa), canonical(ka)
    np.testing.assert_array_equal(np.asarray(qa[..., :head_dim]), np.asarray(q))
    np.testing.assert_array_equal(np.asarray(ka[..., :head_dim]), np.asarray(k))
    np.testing.assert_array_equal(np.asarray(va), np.asarray(v))
    # Segment lanes alone: 0 within a document, -2^14 across documents.
    extra = jnp.einsum(
        "bhsd,bhtd->bhst",
        qa[..., head_dim:].astype(jnp.float32),
        ka[..., head_dim:].astype(jnp.float32),
    )
    same = rel[:, None, :, 0][..., :, None] == rel[:, None, :, 0][..., None, :]
    np.testing.assert_array_equal(
        np.asarray(extra),
        np.broadcast_to(np.where(np.asarray(same), 0.0, -(2.0**14)), extra.shape),
    )

    # Backward: widened gradients (garbage in the segment lanes) give the
    # same input gradients as the plain head lanes.
    gq = jax.random.normal(keys[6], (batch, heads, seq, head_dim)).astype(jnp.bfloat16)
    gk = jax.random.normal(keys[7], (batch, heads, seq, head_dim)).astype(jnp.bfloat16)
    noise = jnp.full((batch, heads, seq, 128 - head_dim), 7.0, jnp.bfloat16)
    bwd = dict(eps=1e-5, q_scale=0.125, block_s=block_s, interpret=True)
    plain = kernel.qk_norm_rope_bwd(xq, xk, wq, wk, cos, sin, gq, gk, v, **bwd)
    widened = kernel.qk_norm_rope_bwd(
        xq, xk, wq, wk, cos, sin,
        canonical(jnp.concatenate([gq, noise], -1)),
        canonical(jnp.concatenate([gk, noise], -1)), v,
        **bwd,
    )
    for got, want in zip(widened, plain):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_split_halves_matches_folded_layout(self):
    """Split-halves q/k (permuted inputs) give the folded layout's scores and
    input gradients after un-permuting."""
    from torchtitan.experiments.tpu.kernels import qk_norm_rope  # pylint: disable=g-import-not-at-top

    batch, seq, heads, head_dim, block_s = 2, 32, 4, 64, 16
    hidden_dim = heads * head_dim
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    xq, xk, xv = (
        jax.random.normal(keys[i], (batch, seq, hidden_dim)).astype(jnp.bfloat16)
        for i in range(3)
    )
    wq = (1.0 + 0.1 * jax.random.normal(keys[3], (hidden_dim,))).astype(jnp.bfloat16)
    wk = (1.0 + 0.1 * jax.random.normal(keys[4], (hidden_dim,))).astype(jnp.bfloat16)
    angles = jax.random.uniform(keys[5], (1, seq, head_dim // 2), minval=-3.0, maxval=3.0)
    emb = jnp.concatenate([angles, angles], axis=-1)
    cos, sin = jnp.cos(emb).astype(jnp.bfloat16), jnp.sin(emb).astype(jnp.bfloat16)
    rel = jnp.array(
        [[0] * 10 + [1] * 12 + [2] * 10, [0] * 20 + [1] * 12], jnp.int32
    )[..., None]
    perm = jnp.asarray(qk_norm_rope.split_halves_permutation(heads, head_dim))
    common = dict(heads=heads, eps=1e-5, q_scale=0.125, block_s=block_s,
                  segment_rel=rel, interpret=True)

    qa, ka, va = kernel.qk_norm_rope_fwd(xq, xk, xv, wq, wk, cos, sin, **common)
    qs, ks, vs = kernel.qk_norm_rope_fwd(
        xq[..., perm], xk[..., perm], xv, wq[perm], wk[perm], cos, sin,
        split_halves=True, **common,
    )
    np.testing.assert_array_equal(np.asarray(vs), np.asarray(va))
    scores = lambda q, k: jnp.einsum(
        "bhsd,bhtd->bhst", q.astype(jnp.float32), k.astype(jnp.float32))
    np.testing.assert_allclose(
        np.asarray(scores(qs, ks)), np.asarray(scores(qa, ka)), atol=0.05
    )

    # Backward: the same per-head gradient, placed in each layout's lanes.
    gq = jax.random.normal(keys[6], (batch, heads, seq, head_dim)).astype(jnp.bfloat16)
    gk = jax.random.normal(keys[7], (batch, heads, seq, head_dim)).astype(jnp.bfloat16)
    lane = np.arange(128)
    dim = lane % 32 + 32 * (lane >= 64)

    def folded(g):  # even heads lanes [0, 64), odd heads [64, 128)
      pad = jnp.zeros_like(g)
      t = jnp.concatenate([g, pad], -1)
      return t.at[:, 1::2].set(jnp.roll(t[:, 1::2], head_dim, axis=-1))

    def split(g):  # head h's values on its own split-halves lanes
      even = (lane % 64) < 32
      out = []
      for h in range(heads):
        own = even if h % 2 == 0 else ~even
        out.append(jnp.where(own, g[:, h][..., dim], 0.0))
      return jnp.stack(out, 1).astype(g.dtype)

    bwd = dict(eps=1e-5, q_scale=0.125, block_s=block_s, interpret=True)
    ref = kernel.qk_norm_rope_bwd(
        xq, xk, wq, wk, cos, sin, folded(gq), folded(gk), va, **bwd)
    got = kernel.qk_norm_rope_bwd(
        xq[..., perm], xk[..., perm], wq[perm], wk[perm], cos, sin,
        split(gq), split(gk), vs, split_halves=True, **bwd)
    inv = jnp.argsort(perm)
    unperm = (got[0][..., inv], got[1][..., inv], got[2], got[3][inv], got[4][inv])
    for name, g, r in zip(("dxq", "dxk", "dxv", "dwq", "dwk"), unperm, ref):
      np.testing.assert_allclose(
          np.asarray(g, np.float32), np.asarray(r, np.float32),
          rtol=2e-2, atol=2e-2, err_msg=name)


if __name__ == "__main__":
  absltest.main()
