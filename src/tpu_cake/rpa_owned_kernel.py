# Adapted from https://github.com/vllm-project/tpu-inference/releases/tag/v0.11.1
# Copyright 2025 The tpu-inference Authors. All rights reserved.

from __future__ import annotations

import inspect
from enum import Enum

import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _cdiv(value: int, divisor: int) -> int:
    if divisor == 0:
        raise ValueError("divisor must be nonzero")
    return (value + divisor - 1) // divisor


def align_to(value: int, alignment: int) -> int:
    return _cdiv(value, alignment) * alignment


def cdiv(value: int, divisor: int) -> int:
    return _cdiv(value, divisor)


def get_dtype_packing(dtype) -> int:
    return 32 // dtypes.itemsize_bits(dtype)


_COMPILER_PARAMS_SUPPORTS_SEMAPHORE = (
    "disable_semaphore_checks" in inspect.signature(pltpu.CompilerParams).parameters
)


def semaphore_kwargs(disable_semaphore_checks: bool) -> dict[str, bool]:
    if _COMPILER_PARAMS_SUPPORTS_SEMAPHORE:
        return {"disable_semaphore_checks": disable_semaphore_checks}
    return {}


class OwnedRpaCase(Enum):
    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self) -> str:
        return {
            OwnedRpaCase.DECODE: "d",
            OwnedRpaCase.PREFILL: "p",
            OwnedRpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution):
        if distribution.shape != (3,):
            raise ValueError("RPA distribution must have three entries")
        if self is OwnedRpaCase.DECODE:
            return 0, distribution[0]
        if self is OwnedRpaCase.PREFILL:
            return distribution[0], distribution[1]
        if self is OwnedRpaCase.MIXED:
            return distribution[1], distribution[2]
        raise ValueError(f"unsupported RPA case: {self}")


def _owned_rpa_kernel(*args, **kwargs):
    # distribution_ref is at index 5 (after kv_lens, page_indices, cu_q_lens,
    # cu_kv_lens, cu_seq_mask_lens).
    distribution_ref = args[5]
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _owned_rpa_kernel_loop(
            seq_idx,
            *args,
            **kwargs,
        )


def _owned_rpa_kernel_loop(
    seq_idx,
    # Prefetch (9 scalar prefetches)
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [flat page indices]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    cu_kv_lens_ref,  # [max_num_seqs + 1]
    cu_seq_mask_lens,  # [max_num_seqs + 1]
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4]
    bkv_update_ids_ref,  # [6]
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    custom_mask_ref,  # [flatten_total_kv_len, head_dim] or None
    zero_mask_ref,  # [bkv_sz, head_dim] or None
    attention_sink_ref,  # [actual_num_kv_heads, num_q_heads_per_kv_head, 128] or None
    relative_states_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head, 128]
    relative_projection_ref,  # [128, 2 * max_kv_len + relative_extent]
    # Output
    o_hbm_ref,  # same shape as q_hbm_ref
    updated_kv_cache_hbm_ref,  # same shape as kv_cache_hbm_ref
    # Scratch
    bkvmask_ref,  # [2, bq_sz, bkv_sz, head_dim] or None
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, ...]
    sems,  # [5, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim]
    *,
    causal: bool = True,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    softmax_dtype: jnp.dtype | None = None,
    relative_extent: int | None = None,
    static_q_len: int | None = None,
    bq_sz,  # bq fetch size
    bkv_sz,  # bkv prefetch size
    bq_csz,  # bq compute size
    bkv_csz,  # bkv compute size
    case: OwnedRpaCase = OwnedRpaCase.MIXED,
    skip_kv_mask: bool = False,
    tpu_version: int = 6,
    debug_mode: bool = False,
    mask_aligned_to_cu_kv: bool = False,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]

    use_causal_mask = causal
    if case == OwnedRpaCase.DECODE:
        use_causal_mask = False

    out_dtype = acc_ref.dtype
    (
        actual_num_kv_heads,
        _max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        _total_num_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    num_page_indices = page_indices_ref.shape[0]
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    assert bkv_sz % page_size == 0
    assert bkv_sz % bkv_csz == 0, f"bkv_sz={bkv_sz} not divisible by bkv_csz={bkv_csz}"
    bkv_p = bkv_sz // page_size
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0

    if sliding_window is not None:
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_start = cu_q_lens_ref[next_seq_idx]
        next_q_end = cu_q_lens_ref[next_seq_idx + 1]
        next_q_len = next_q_end - next_q_start
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_kv_q_gap = next_kv_len - next_q_len
        next_seq_start_bkv_idx = jnp.maximum(next_kv_q_gap - sliding_window, 0) // bkv_sz

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_csz, head_dim]
        v,  # [bkv_csz, head_dim]
        l_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        m_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
        xai_temperature_reg=None,
        custom_mask_data=None,
        relative_state=None,
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        actual_bq_csz = q.shape[0] // num_q_heads_per_kv_head
        assert k.shape == (bkv_csz, head_dim)
        assert v.shape == (bkv_csz, head_dim)
        assert l_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert m_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert k.dtype == v.dtype

        # Follow FlashAttention-2 forward pass.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        s = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)

        s_scale = sm_scale
        if k_scale is not None:
            s_scale *= k_scale
        if q_scale is not None:
            s_scale *= q_scale

        s *= s_scale

        if relative_state is not None:
            assert relative_state.shape == (
                actual_bq_csz * num_q_heads_per_kv_head,
                relative_projection_ref.shape[0],
            )
            projection_padding = (relative_projection_ref.shape[1] - relative_extent) // 2
            relative_bias_rows = []
            for query_offset in range(actual_bq_csz):
                query_position = processed_q_len + query_offset
                projection_start = (
                    projection_padding + relative_extent - 1 - query_position + processed_kv_len
                )
                aligned_projection_start = pl.multiple_of(
                    projection_start - jnp.mod(projection_start, 128),
                    128,
                )
                projection_offset = projection_start - aligned_projection_start
                projection_window = relative_projection_ref[
                    :, pl.ds(aligned_projection_start, bkv_csz + 128)
                ]
                state_start = query_offset * num_q_heads_per_kv_head
                query_relative_state = relative_state[
                    state_start : state_start + num_q_heads_per_kv_head, :
                ]
                relative_bias_window = pl.dot(
                    query_relative_state.astype(jnp.float32),
                    projection_window,
                )
                relative_bias_window = pltpu.roll(
                    relative_bias_window,
                    bkv_csz + 128 - projection_offset,
                    axis=1,
                )
                relative_bias = relative_bias_window[:, :bkv_csz]
                output_start = jnp.maximum(query_position - relative_extent + 1, 0)
                relative_positions = processed_kv_len + jnp.arange(bkv_csz)
                relative_valid = (relative_positions >= output_start) & (
                    relative_positions <= query_position
                )
                relative_bias_rows.append(jnp.where(relative_valid[None, :], relative_bias, 0.0))
            s += jnp.concatenate(relative_bias_rows, axis=0)

        # xai temperature scaling
        if xai_temperature_reg is not None:
            s = s * xai_temperature_reg[:, None]

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        # Use int16 for span computations when safe: non-f32 dtype on TPU v6+
        # with causal mask. Custom mask shapes can trigger a Mosaic compiler bug.
        int_ty = jnp.int32
        if get_dtype_packing(q.dtype) != 1 and tpu_version >= 6 and use_causal_mask:
            int_ty = jnp.int16
        processed_q_len_int = processed_q_len.astype(int_ty)
        processed_kv_len_int = processed_kv_len.astype(int_ty)
        effective_kv_len_int = effective_kv_len.astype(int_ty)
        q_span = processed_q_len_int + (
            lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        ).astype(int_ty)
        k_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, s.shape, 1)
        v_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, v.shape, 0)

        mask = None
        if use_causal_mask:
            assert not skip_kv_mask
            mask = mask_and(mask, q_span >= k_span)
        elif custom_mask_data is not None:
            # custom_mask_data: [actual_bq_csz, bkv_csz] int32, 1=keep
            custom_mask_expanded = jnp.repeat(custom_mask_data, num_q_heads_per_kv_head, axis=0)
            mask = mask_and(mask, custom_mask_expanded == 1)

        if not skip_kv_mask:
            mask = mask_and(mask, k_span < effective_kv_len_int)
            v = jnp.where(v_span < effective_kv_len_int, v, 0.0)

        if sliding_window is not None:
            mask = mask_and(mask, q_span < k_span + sliding_window)

        if mask is not None:
            s = jnp.where(mask, s, mask_value)

        if softmax_dtype is not None:
            s = s.astype(softmax_dtype)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = m_ref[...].astype(jnp.float32)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_ref[...] = m_curr.astype(out_dtype)
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_ref[...].astype(jnp.float32)
        l_ref[...] = (exp_m_diff * l_prev + p_rowsum).astype(out_dtype)

        return p, v, exp_m_diff

    def flash_attention_step2_pv(
        p,  # [actual_bq_csz * num_q_heads_per_kv_head, bkv_csz]
        v,  # [bkv_csz, head_dim]
        exp_m_diff,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        o_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
    ):
        assert len(p.shape) == 2
        assert p.shape[0] % num_q_heads_per_kv_head == 0
        assert p.shape[1] == bkv_csz
        actual_bq_csz = p.shape[0] // num_q_heads_per_kv_head
        assert v.shape == (bkv_csz, head_dim)
        assert exp_m_diff.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert o_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, head_dim)
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
        o_prev = o_ref[...].astype(jnp.float32)
        o_ref[...] = (broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv).astype(out_dtype)

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, *, wait=False):
        if custom_mask_ref is None:
            return
        sem = sems.at[4, bkvmask_sem_idx]
        kvmask_vmem_ref = bkvmask_ref.at[bkvmask_sem_idx]

        if mask_aligned_to_cu_kv:
            # Host padded each mask row to the page-aligned kv_len (= cu_kv_lens
            # delta), so stride/offset/size are statically tiling(8)-divisible.
            mask_kv_len = pl.multiple_of(cu_kv_lens_ref[seq_idx + 1] - cu_kv_lens_ref[seq_idx], 8)
        else:
            mask_kv_len = kv_lens_ref[seq_idx]
        mask_start = bkvmask_idx * bkv_sz
        mask_left = mask_kv_len - mask_start
        load_kvmask_sz = jnp.minimum(bkv_sz, mask_left)
        if mask_aligned_to_cu_kv:
            load_kvmask_sz = pl.multiple_of(load_kvmask_sz, 8)

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        load_q_sz = jnp.minimum(bq_sz, q_end - q_len_start)

        cur_seq_mask_start = cu_seq_mask_lens[seq_idx]
        cur_bq_mask_start = cur_seq_mask_start + bq_idx * bq_sz * mask_kv_len
        zero_sz = bkv_sz - load_kvmask_sz
        if mask_aligned_to_cu_kv:
            cur_seq_mask_start = pl.multiple_of(cur_seq_mask_start, 8)
            zero_sz = pl.multiple_of(zero_sz, 8)

        def loop_body(i, _):
            start = cur_bq_mask_start + i * mask_kv_len + mask_start
            if mask_aligned_to_cu_kv:
                start = pl.multiple_of(start, 8)
            _async_copy(
                custom_mask_ref.at[pl.ds(start, load_kvmask_sz)],
                kvmask_vmem_ref.at[i, pl.ds(0, load_kvmask_sz)],
                sem,
                wait,
            )
            _async_copy(
                zero_mask_ref.at[pl.ds(0, zero_sz)],
                kvmask_vmem_ref.at[i, pl.ds(load_kvmask_sz, zero_sz)],
                sem,
                wait,
            )

        lax.fori_loop(0, load_q_sz, loop_body, None, unroll=False)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        # sglang-jax: use cu_kv_lens for page_indices offset.
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            for i in range(bkv_p):
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(page_indices_ref[page_idx] * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )

            new_kv_len_start = q_end - kv_left_frm_new
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        # sglang-jax: use cu_kv_lens for page_indices offset.
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)

            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size + ignore,
                        sz,
                    )
                ],
                sem,
                wait,
            )
            return update_sz - sz, 0

        if not wait:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx):
        return _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)

    def wait_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx):
        return _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def strided_load(ref, start, sz, step, *, dtype=None):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        r, l = ref.shape
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        vec = jnp.concat([ref[pl.ds(start + i, sz // step, step)] for i in range(folds)], axis=1)
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def strided_store(ref, start, sz, step, val):
        assert get_dtype_packing(ref.dtype) == 1
        assert ref.dtype == val.dtype
        assert ref.shape == val.shape
        assert len(ref.shape) == 2
        r, l = ref.shape
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        for i in range(folds):
            ref[pl.ds(start + i, sz // step, step)] = val[:, i * 128 : (i + 1) * 128]

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        start *= num_q_heads_per_kv_head_per_packing
        sz *= num_q_heads_per_kv_head_per_packing
        return strided_load(q_ref, start, sz, 1, dtype=q_dtype)

    def load_bkv(bkv_sem_idx, kv_head_idx, start, sz):
        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = bkv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(bkv_sz * step, head_dim)

        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return k, v

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        k = pltpu.bitcast(k.astype(repack_ty), kv_dtype)
        v = pltpu.bitcast(v.astype(repack_ty), kv_dtype)
        return k, v

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[
            ..., : shape[-1]
        ]

    def mask_and(mask, new_mask):
        if mask is None:
            return new_mask
        return jnp.logical_and(mask, new_mask)

    def process(static_q_len=None):
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        actual_bq_csz = min(bq_csz, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            next_bq_start_bkv_idx = 0
            if sliding_window is not None:
                next_bq_start_bkv_idx = (
                    jnp.maximum(kv_q_gap + (bq_idx + 1) * actual_bq_sz - sliding_window, 0)
                    // bkv_sz
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx, next_bkv_idx)
            next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx, next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            # Initialize l, m before bkv loop.
            if attention_sink_ref is not None:
                # Attention sink: m = sink logits, l = 1.0
                # (pretend we've already seen a virtual token with logit = sink_value).
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
                l_ref[...] = jnp.full_like(l_ref, 1.0)
                for kv_head_idx in range(actual_num_kv_heads):
                    sinks = attention_sink_ref[kv_head_idx]  # [num_q_heads_per_kv_head, 128]
                    lm_start = 0
                    lm_size = actual_bq_sz * num_q_heads_per_kv_head
                    sink_tiled = jnp.tile(sinks, (actual_bq_sz, 1))
                    m_ref.at[kv_head_idx, pl.ds(lm_start, lm_size)][...] = sink_tiled.astype(
                        out_dtype
                    )
            else:
                l_ref[...] = jnp.full_like(l_ref, 0.0)
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = 0
            if sliding_window is not None:
                start_bkv_idx = jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz
            if use_causal_mask:
                effective_kv_len = jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
            else:
                effective_kv_len = kv_len
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            # xai temperature computation
            xai_temperature_reg = None
            if xai_temperature_len is not None:
                prefix_len = kv_len - q_len
                local_q_offset = (
                    bq_idx * bq_sz
                    + lax.iota(jnp.int32, actual_bq_sz * num_q_heads_per_kv_head)
                    // num_q_heads_per_kv_head
                )
                absolute_q_position = prefix_len + local_q_offset
                xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
                _qtemp = jnp.log2(absolute_q_position.astype(jnp.float32)) * xai_temperature_scale
                xai_temperature_reg = jnp.where(
                    absolute_q_position > xai_temperature_len, _qtemp, 1.0
                )

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                assert bkv_sz % kv_packing == 0

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                )
                processed_kv_len = bkv_idx * bkv_sz

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)
                    if custom_mask_ref is not None:
                        start_fetch_mask(next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                # Wait for custom mask if applicable
                if custom_mask_ref is not None:
                    wait_fetch_mask(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in last bq loop.
                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                if debug_mode:
                    return

                # Load custom mask data for this block
                custom_mask_data = None
                if bkvmask_ref is not None:
                    custom_mask_data = bkvmask_ref[bkv_sem_idx, :actual_bq_sz, :, 0]

                # Flash attention with cur bkv and bq
                effective_bkv_sz = jnp.minimum(effective_kv_len - bkv_idx * bkv_sz, bkv_sz)
                effective_bkv_sz = jnp.maximum(effective_bkv_sz, 0)

                # Use static loop bound to avoid potential Pallas issues with
                # dynamic loop bounds. The @pl.when guard skips invalid iterations.
                max_num_loops = bkv_sz // bkv_csz

                @pl.loop(0, max_num_loops, unroll=False)
                def attention_loop(idx):
                    bkv_start = idx * bkv_csz

                    @pl.when(bkv_start < effective_bkv_sz)
                    def _():
                        for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                            # Slice custom mask for this compute sub-block
                            cur_mask_data = None
                            if custom_mask_data is not None:
                                cur_mask_data = bkvmask_ref[
                                    bkv_sem_idx,
                                    pl.ds(bq_start, actual_bq_csz),
                                    pl.ds(bkv_start, bkv_csz),
                                    0,
                                ]

                            # Slice xai temperature for this compute sub-block
                            cur_xai_temp = None
                            if xai_temperature_reg is not None:
                                q_head_start = bq_start * num_q_heads_per_kv_head
                                q_head_sz = actual_bq_csz * num_q_heads_per_kv_head
                                cur_xai_temp = xai_temperature_reg[
                                    q_head_start : q_head_start + q_head_sz
                                ]

                            for kv_head_idx in range(actual_num_kv_heads):
                                bk_c, bv_c = load_bkv(
                                    bkv_sem_idx,
                                    kv_head_idx,
                                    bkv_start,
                                    bkv_csz,
                                )
                                bq_c = load_bq(bq_sem_idx, kv_head_idx, bq_start, actual_bq_csz)

                                lm_slice_start = bq_start * num_q_heads_per_kv_head
                                lm_slice_size = actual_bq_csz * num_q_heads_per_kv_head
                                lm_slice = (
                                    kv_head_idx,
                                    pl.ds(lm_slice_start, lm_slice_size),
                                )

                                cur_p, cur_v, cur_exp_m_diff = flash_attention_step1_qk_softmax(
                                    bq_c,
                                    bk_c,
                                    bv_c,
                                    l_ref.at[*lm_slice],
                                    m_ref.at[*lm_slice],
                                    processed_q_len=processed_q_len + bq_start,
                                    processed_kv_len=processed_kv_len + bkv_start,
                                    effective_kv_len=effective_kv_len,
                                    xai_temperature_reg=cur_xai_temp,
                                    custom_mask_data=cur_mask_data,
                                    relative_state=(
                                        relative_states_ref[
                                            kv_head_idx,
                                            pl.ds(
                                                q_start + bq_idx * actual_bq_sz + bq_start,
                                                actual_bq_csz,
                                            ),
                                            :,
                                            :,
                                        ].reshape(
                                            actual_bq_csz * num_q_heads_per_kv_head,
                                            -1,
                                        )
                                        if relative_states_ref is not None
                                        else None
                                    ),
                                )
                                flash_attention_step2_pv(
                                    cur_p,
                                    cur_v,
                                    cur_exp_m_diff,
                                    acc_ref.at[*lm_slice],
                                )

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)
            out = (
                acc * pl.reciprocal(l, approx=True)
                if (l.dtype == jnp.float32 and out_dtype != jnp.float32)
                else lax.div(acc, l)
            ).astype(out_dtype)

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            out_ref = (
                bo_x2_ref.at[bo_sem_idx]
                .bitcast(jnp.int32)
                .reshape(
                    actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head_per_packing,
                    head_dim,
                )
            )
            out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
            strided_store(out_ref, 0, out_ref.shape[0], 1, out)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(seq_idx=start_seq_idx, bq_idx=0, bq_sem_idx=0)
        bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32)
        zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)
        bkv_x2_int32_ref[0] = zeros
        start_fetch_bkv(seq_idx=start_seq_idx, bkv_idx=cur_seq_start_bkv_idx, bkv_sem_idx=0)
        bkv_x2_int32_ref[1] = zeros
        if custom_mask_ref is not None:
            start_fetch_mask(start_seq_idx, 0, 0, 0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process(static_q_len=static_q_len)

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)

    ### ------- Kernel end ------- ###


def owned_rpa_has_bank_conflicts(stride, distance=24, num_banks=32):
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False
