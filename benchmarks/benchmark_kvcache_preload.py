#!/usr/bin/env python3
"""Measure v2 dense KV-cache FAInfer kernel time around QK L1 preload boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

from benchmark_utils import _load_kernels, _select_kernel, build_profiler, write_results

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Build label, such as baseline or preload")
    parser.add_argument(
        "--profile-root", type=Path, required=True, help="New directory for profiler traces"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--json", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-check", action="store_true", help="Skip sampled FP32 correctness check"
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 1:
        parser.error("--warmup and --repeat must be positive")
    if args.profile_root.exists():
        parser.error(f"profile directory already exists: {args.profile_root}")
    return args


def run_case(seq_kv, args, device):
    from flash_attn_npu import flash_attn_with_kvcache

    batch, seq_q, heads, dim = 512, 32, 8, 128
    torch.manual_seed(args.seed + seq_kv)
    query = torch.randn((batch, seq_q, heads, dim), device=device, dtype=torch.float16)
    key = torch.randn((batch, seq_kv, heads, dim), device=device, dtype=torch.float16)
    value = torch.randn((batch, seq_kv, heads, dim), device=device, dtype=torch.float16)
    cache_seqlens = torch.full((batch,), seq_kv, device=device, dtype=torch.int32)

    def forward():
        return flash_attn_with_kvcache(
            query,
            key,
            value,
            cache_seqlens=cache_seqlens,
            causal=False,
            num_splits=1,
        )

    max_abs_error = None
    if not args.skip_check:
        output = forward()
        torch.npu.synchronize()
        sample_indices = [0, batch // 2, batch - 1]
        query_ref = query[sample_indices].float().cpu().transpose(1, 2)
        key_ref = key[sample_indices].float().cpu().transpose(1, 2)
        value_ref = value[sample_indices].float().cpu().transpose(1, 2)
        scores = torch.matmul(query_ref, key_ref.transpose(-1, -2)) * (dim**-0.5)
        reference = torch.matmul(torch.softmax(scores, dim=-1), value_ref).transpose(1, 2)
        output_sample = output[sample_indices].float().cpu()
        max_abs_error = (output_sample - reference).abs().max().item()
        try:
            torch.testing.assert_close(output_sample, reference, atol=0.05, rtol=0.05)
        except AssertionError as exc:
            raise AssertionError(
                f"Skv={seq_kv} sampled output differs from FP32 reference"
            ) from exc

    for _ in range(args.warmup):
        forward()
    torch.npu.synchronize()

    case_profile = args.profile_root / f"skv_{seq_kv}"
    with build_profiler(case_profile, warmup=2, repeat=args.repeat) as profiler:
        for _ in range(2 + args.repeat):
            forward()
            torch.npu.synchronize()
            profiler.step()
        profiler.step()

    kernels, kernel_csv = _load_kernels(case_profile)
    hits = _select_kernel(kernels, re.compile(r"FAInfer", re.I))
    if len(hits) != args.repeat:
        names = sorted({row.get("Name", "") for row in kernels})
        raise RuntimeError(
            f"Skv={seq_kv}: expected {args.repeat} FAInfer kernels, found {len(hits)}; "
            f"kernel CSV={kernel_csv}; names={names[:30]}"
        )
    durations_us = [duration_ms * 1000 for _, duration_ms in hits]
    return {
        "label": args.label,
        "group": "small" if seq_kv < 64 else "boundary" if seq_kv < 256 else "preload",
        "batch": batch,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "heads_q": heads,
        "heads_kv": heads,
        "head_dim": dim,
        "dtype": "fp16",
        "layout": "BSND",
        "fa_kernel_mean_us": sum(durations_us) / len(durations_us),
        "fa_kernel_min_us": min(durations_us),
        "fa_kernel_max_us": max(durations_us),
        "fa_kernel_count": len(hits),
        "sample_max_abs_error": max_abs_error,
        "kernel_csv": str(kernel_csv),
    }


def main():
    sys.path.insert(0, str(REPO_ROOT))
    from flash_attn_npu.flash_attn_npu_interface import flash_attn_npu as extension

    args = parse_args()
    expected_extension = REPO_ROOT / "flash_attn_npu"
    extension_path = Path(extension.__file__).resolve()
    if extension_path.parent != expected_extension:
        raise RuntimeError(
            f"Imported extension from {extension_path}, expected {expected_extension}"
        )
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    print(f"device={torch.npu.get_device_name(args.device)} extension={extension_path}", flush=True)

    rows = []
    for seq_kv in (32, 63, 128, 129, 256, 512):
        row = run_case(seq_kv, args, device)
        rows.append(row)
        print(
            f"{args.label} Skv={seq_kv}: {row['fa_kernel_mean_us']:.2f} us "
            f"(sample max abs error={row['sample_max_abs_error']})",
            flush=True,
        )

    write_results(
        rows,
        str(args.out),
        str(args.json),
        {
            "label": args.label,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "device": torch.npu.get_device_name(args.device),
            "extension": str(extension_path),
            "metric": "FAInfer device kernel duration (us)",
        },
    )
    print(f"results: {args.out} {args.json}", flush=True)


if __name__ == "__main__":
    main()
