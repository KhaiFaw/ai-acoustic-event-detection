# Firmware integration notes

This directory contains the project-specific application layer and configuration snapshots used for the Renesas RA8P1 Titan Board.

It is not a complete copy of the RT-Thread SDK. Generated build output, IDE caches, large board-support trees, and installed packages are intentionally excluded from this public portfolio repository.

## Expected environment

- RT-Thread Studio
- Titan Board SDK/BSP
- Renesas RA Flexible Software Package compatible with the project configuration
- TensorFlow Lite for Microcontrollers package
- Arm CMSIS-DSP

## Layout

```text
firmware/
├── src/
│   ├── hal_entry.c
│   ├── cpp_test.cpp
│   ├── sound_model.h
│   ├── mfcc_norm_params.h
│   ├── lcd_alert_ui.c
│   ├── lcd_alert_ui.h
│   └── cmsis_dsp_fft/
├── project-config/
│   ├── configuration.xml
│   ├── rt-thread.config
│   └── rtconfig.h
└── SConscript
```

## Integration outline

1. Create or open a Titan Board BSP project in RT-Thread Studio.
2. Back up the local project before changing generated configuration.
3. Copy the files from `firmware/src/` into the BSP project’s `src/` directory.
4. Use `project-config/configuration.xml` as the FSP configuration reference.
5. Restore the required RT-Thread/TensorFlow Lite Micro package selections.
6. Confirm the MAX9814 is connected to ADC0 and that the ADC channel matches the source configuration.
7. Regenerate platform files through the supported RT-Thread/FSP workflow.
8. Build, flash, and monitor the serial terminal.

## Important notes

- The model and normalization parameters must stay paired.
- The model’s input tensor shape is `1 × 39 × 61 × 1`.
- Generated FSP files can differ between tool versions.
- The LCD interface requires the Titan Board display stack and the matching GLCDC/MIPI configuration.
- Do not copy local toolchain paths, credentials, or IDE caches into a public repository.
