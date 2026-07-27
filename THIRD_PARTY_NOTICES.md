# Third-party notices

This repository contains project-specific application code that integrates with several third-party components.

## CMSIS-DSP

Selected FFT implementation files are included under `firmware/src/cmsis_dsp_fft/`. Their source headers identify them as Arm CMSIS-DSP code licensed under Apache License 2.0. The accompanying licence text is preserved at `third_party/CMSIS-LICENSE.txt`.

## RT-Thread

RT-Thread is required by the firmware but the full RT-Thread source tree is not included in this curated repository. Refer to the upstream RT-Thread project and its licence terms.

## TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is required by `cpp_test.cpp` but its package source is not included here. Refer to the upstream TensorFlow project and its licence terms.

## Renesas Flexible Software Package

The project targets the Renesas RA8P1 and uses Renesas FSP-generated configuration. The full Renesas FSP source tree is not included. Refer to the upstream Renesas FSP distribution and the licence headers in the version you use.

## Trained model

The generated model header is included for demonstration and reproducibility. Dataset licences and usage restrictions must be considered before redistributing any original training audio.
