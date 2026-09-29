# Verification and evidence boundaries

Updated 30 September 2026. The owner authorised merging the software-verified correction without physical-board acceptance. Hardware checks below remain future validation work, not completed tests.

## Artifact results

Sixteen host tests pass, including malformed-metadata regressions. The 117,136-byte FlatBuffer is TFLite schema v3 with one INT8 input `[1,39,61,1]`, one INT8 output `[1,1,1,6]`, and six builtin operator types supported by the resolver. Both normalization arrays contain 2,379 finite entries; standard deviations are positive.

Model SHA-256: `59f72961fe2ff328dfc4eca839c3479d22d6dfa795b512ccb9d537559343a2ef`.

The audit found input scale `0.05333127826452255` in the model versus `0.088802` in `hal_entry.c`. The correction uses `0.053331278f`; a regression test explicitly rejects the previous value. This changes feature quantization. Physical-board validation has not been performed, so historical accuracy must not be assigned to this revised firmware.

The output scale is `1/256`, consistent with the integer confidence conversion. Checks require scalar finite-positive quantization parameters, INT8-range zero points, exactly one input/output, valid tensor/operator-code indices and a SOFTMAX output producer. The checker also rejects missing/truncated model declarations, feature-count mismatches, non-finite means and zero standard deviations.

## Local firmware build

A fresh full-BSP compile and link passed on 30 September 2026 with Arm GNU 13.3.Rel1 / GCC 13.3.1 and GNU Make 4.2.1. The corrected application was overlaid onto an isolated copy of the existing local BSP. No original workspace files were overwritten, and no device was flashed. See [software validation](software-validation.md) for the command, artefact hashes and limitations.

## What these checks cannot establish

No board inference, acoustic recording, latency, power or runtime total-memory measurement was performed. A successful compile/link is not proof of board behaviour. The training script named in the generated header, raw evaluation windows and complete BSP are absent from the public snapshot. The paired model and normalization arrays have compatible dimensions, but those checks cannot prove their training provenance or feature-by-feature parity.

## Future owner-run board acceptance

1. Record RT-Thread Studio, BSP, FSP, TFLM and compiler versions, commit and physical ADC pin.
2. Build from a fresh BSP and record flash/RAM output; do not infer total RAM from the tensor arena.
3. Record microphone settings, distance, room, playback source and capture rate.
4. Keep raw classifier predictions and filtered alert decisions separately. Record `CLEAN_BACKGROUND_OUTPUT_MODE`, quiet threshold and loop delay.
5. Compare quantized feature bytes and output scores against the exact training pipeline for a fixed consented sample.
6. Evaluate a balanced set across all six classes, preserving window-level truth/prediction records and showing the confusion matrix.
7. Exercise LCD and serial agreement, background, low-confidence, repeated-hit and cooldown behavior.

Do not distribute household recordings without consent or training material without appropriate rights.
