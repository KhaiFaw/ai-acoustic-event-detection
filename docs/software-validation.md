# Software validation record

30 September 2026. This record covers the corrected application source in this repository. Physical-board testing was explicitly excluded from this maintenance acceptance; no acoustic-performance claim is made.

## Host checks

- Python 3.12.14, using the versions pinned in `requirements-checks.txt`.
- `python tools/check_model_contract.py`: passes.
- `python -m unittest discover -s tests -v`: all 16 tests pass.
- One INT8 input `[1,39,61,1]`, scale `0.05333127826452255`, zero point `3`.
- One INT8 output `[1,1,1,6]`, scale `0.00390625`, zero point `-128`, produced by SOFTMAX.
- Firmware input scale `0.053331278f` matches the model; the previous `0.088802` value is rejected by a regression test.
- Both normalization arrays have 2,379 finite values, with positive standard deviations.

These checks inspect the deployed artifacts; they do not run the CNN or establish training provenance.

## Fresh full-BSP build

The existing local Titan-board project was copied to a separate build area. The repository's `firmware/src/` application source was overlaid there. Generated Makefiles were copied without old objects or dependency files, and absolute paths were retargeted to the isolated copy. An external TFLM include reference was replaced with the staged package after confirming all 140 referenced header files matched by SHA-256. The original project was not changed.

| Item | Recorded result |
|---|---|
| Compiler | Arm GNU 13.3.Rel1, GCC 13.3.1 |
| Build driver | GNU Make 4.2.1 |
| SDK declarations | Renesas FSP 6.0.0; RT-Thread 5.1.0 |
| TFLM project baseline | `520f8f1e8193eef516e30e1951c885e001896c12` |
| Build command | `make -B -j4 all` in the staged `Debug` directory |
| Exit status | `0` - full compile, link, binary conversion and size output complete |
| Rebuilt objects | All 266 objects, including `hal_entry.c`, `cpp_test.cpp` and `lcd_alert_ui.c` |
| Compiler/linker diagnostics | 0 errors, 62 warnings |
| Binary size | 404,916 bytes |
| ELF file size | 7,127,684 bytes, including debug information |
| ELF section summary | `text=404344`, `data=572`, `bss=34792092` bytes |
| Linker flash allocation | 425,984 / 1,048,576 configured bytes |
| Linker internal-RAM allocation | 1,245,184 / 1,523,712 configured bytes, including alignment |
| External no-init reservation | 33,554,432 bytes in OSPI1_CS0 |

The section summary is a linker accounting result, **not measured runtime memory use**. The BSS figure includes a 32 MiB external no-init reservation and must not be presented as internal MCU RAM consumption or peak runtime usage. Source configuration snapshots match the staged build configuration, ignoring line-ending differences where applicable.

Artifact SHA-256 values:

```text
rtthread.elf  177ba4a9eca1c16a1c513ee8007394eacecc7fc2b8d3b536790ac2fc9eab043b
rtthread.bin  25dfa852617e22f1609577c1b2dabda7f7dedc62f5343665e90e35ede747c4e1
sound_model  59f72961fe2ff328dfc4eca839c3479d22d6dfa795b512ccb9d537559343a2ef
```

Build output and the complete BSP remain local; this public repository is still an application-layer snapshot rather than a standalone SDK checkout.

## Build qualifications

- The existing generated build configuration was retained. Compile commands target ARMv8.1-M, while the existing link command specifies `-mcpu=cortex-m4`. The link succeeded, but the target/multilib settings should be reviewed against the Titan-board configuration before any flash or hardware acceptance. This maintenance did not silently change them.
- The build emits warnings, including unused legacy application helpers and RT-Thread scheduler implicit declarations. A passing build is not a warning-free claim.
- FSP/RT-Thread versions are read from local source headers and the TFLM baseline from its project metadata. These identify the local build inputs; they are not a claim that a pristine SDK checkout or a fresh IDE import was reproduced. Full dependencies are not redistributed here.
- No flash, board boot, LCD/ADC exercise, inference, feature parity, audio accuracy, latency, power or runtime-memory test was performed.
- The historical 14/16-window result remains historical and is not assigned to this correction.

See [verification boundaries and future board procedure](verification.md).
