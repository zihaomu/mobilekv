# Format Descriptor

`mobilekv` stores KV cache bytes and layout metadata. It does not run quantization or dequantization math.

## Purpose

`FormatDescriptor` is attached to `TemplateConfig` so each K/V template can describe how bytes should be interpreted by external kernels or runtime code.

## Fields

- `quant_scheme`: `None`, `PerTensorAffine`, `PerChannelSymmetric`, `PerGroupAffine`
- `group_size`: group width for grouped schemes (`0` means not used)
- `storage_type`: scalar type used for KV payload bytes (`FP16`, `INT8`, etc.)
- `scale_type`: scalar type of scales if required by the scheme
- `has_zero_point`: whether quantized values use zero-point
- `zero_point_type`: scalar type used for zero-point storage

## Current Behavior

- Plain/Packed templates initialize:
- `quant_scheme = None`
- `storage_type = template scalar type`
- `has_zero_point = false`
- All fields are queryable through `templ.config().format`.

## Non-Goals

- No quant/dequant compute kernels.
- No automatic scale/zero-point allocation.
- No coupling to specific attention kernels.

