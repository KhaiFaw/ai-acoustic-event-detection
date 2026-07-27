#include <rtthread.h>
#include "hal_data.h"
#include "sound_model.h"
#include "mfcc_norm_params.h"
#include "lcd_alert_ui.h"
#include "arm_math.h"
#include <stdint.h>
#include <math.h>


#define AUDIO_SAMPLE_COUNT 16000
#define TARGET_SAMPLE_RATE_HZ 16000

#define MIC_ADC_CHANNEL ADC_CHANNEL_0

#define SAMPLE_PERIOD_US 52
#define ADC_SETTLE_US    5

#define TENSOR_ARENA_SIZE (180 * 1024)

/*
 * Model input shape is:
 * 1 x 39 x 61 x 1
 */
#define FEATURE_COEFFS 39
#define FEATURE_FRAMES 61
#define FEATURE_LEN    (FEATURE_COEFFS * FEATURE_FRAMES)

/*
 * Frame settings:
 * 400 samples = about 25 ms at 16 kHz
 * 256 samples = about 16 ms hop
 */
#define FRAME_SIZE  512
#define FRAME_STEP  256
#define FRAME_COUNT 61

#define FFT_SIZE      512
#define FFT_BIN_COUNT ((FFT_SIZE / 2) + 1)

#define MEL_FILTER_COUNT 40
#define MEL_FMIN_HZ      0.0f
#define MEL_FMAX_HZ      8000.0f

#define MFCC_COEFF_COUNT 13

#define MODEL_INPUT_SCALE       0.088802f
#define MODEL_INPUT_ZERO_POINT  (3)

/* ============================================================
 * Final demo settings
 * ------------------------------------------------------------
 * Set FINAL_DEMO_MODE to 1 for clean serial output.
 * The detailed FFT/MFCC debug functions are kept in this file,
 * but the final demo loop below uses silent helper functions.
 * ============================================================ */
#define FINAL_DEMO_MODE          0
#define FINAL_DEMO_DELAY_MS      1500

/* ============================================================
 * Audio dump collection mode
 * ------------------------------------------------------------
 * Set AUDIO_DUMP_COLLECTION_MODE to 1 only when you want to
 * collect/screenshot centered ADC samples from the MAX9814 mic.
 * Set it back to 0 for the final demo inference mode.
 * ============================================================ */
#define AUDIO_DUMP_COLLECTION_MODE 0
#define AUDIO_DUMP_START_DELAY_MS  2000
#define AUDIO_DUMP_GAP_MS          3000

/* ============================================================
 * Clean background output mode for viva demo recording
 * ------------------------------------------------------------
 * Purpose:
 * - If the prediction confidence is below its alert threshold,
 *   display it as Unknown/Background instead of showing the raw class.
 * - If the signal is very quiet, also display it as Unknown/Background.
 *
 * IMPORTANT:
 * This is only for clean viva/demo output.
 * Turn this OFF later for real testing/final accurate version.
 * ============================================================ */
#define CLEAN_BACKGROUND_OUTPUT_MODE 1

/*
 * In your background-only test, p2p was around 4000-5700.
 * So 6500 is used as a quiet/background gate.
 * If real target sounds become hidden, lower this value.
 */
#define DEMO_QUIET_P2P_THRESHOLD 6500

/* ============================================================
 * Alert decision logic
 * ------------------------------------------------------------
 * Purpose:
 * 1) Ignore low-confidence predictions as "No reliable sound".
 * 2) Require repeated detection for stable sounds to reduce false alerts.
 * 3) Apply cooldown so the same alert is not printed repeatedly.
 *
 * Class index mapping:
 * 0 Baby Crying
 * 1 Glass Breaking
 * 2 Alarm/Siren
 * 3 Door Knock
 * 4 Doorbell
 * 5 Unknown/Background
 * ============================================================ */
#define ALERT_CLASS_COUNT       6
#define UNKNOWN_CLASS_INDEX     5
#define ALERT_DISABLED_TH       101
#define ALERT_COOLDOWN_MS       5000
#define ALERT_RELEASE_MISSES    2

static const char * alert_class_names[ALERT_CLASS_COUNT] =
{
    "Baby Crying",
    "Glass Breaking",
    "Alarm/Siren",
    "Door Knock",
    "Doorbell",
    "Unknown/Background"
};

/*
 * Thresholds are class-specific because live testing showed:
 * - Doorbell, Alarm/Siren, and Baby Crying are reliable at high confidence.
 * - Glass Breaking is more transient, so a lower threshold is allowed.
 * - Door Knock is currently weak, so a higher threshold is used.
 * - Unknown/Background is never used to trigger an alert.
 */
static const int alert_threshold_percent[ALERT_CLASS_COUNT] =
{
    75,                 /* Baby Crying */
    55,                 /* Glass Breaking */
    75,                 /* Alarm/Siren */
    80,                 /* Door Knock */
    75,                 /* Doorbell */
    ALERT_DISABLED_TH   /* Unknown/Background */
};

/*
 * Required consecutive hits before triggering alert.
 * Glass Breaking is allowed with 1 hit because it is usually a short sound.
 * Doorbell, Alarm/Siren, Baby Crying and Door Knock need 2 consecutive hits.
 */
static const int alert_required_hits[ALERT_CLASS_COUNT] =
{
    2,  /* Baby Crying */
    1,  /* Glass Breaking */
    2,  /* Alarm/Siren */
    2,  /* Door Knock */
    2,  /* Doorbell */
    0   /* Unknown/Background */
};

static int alert_hit_count[ALERT_CLASS_COUNT] = {0};
static rt_tick_t alert_last_tick[ALERT_CLASS_COUNT] = {0};
static int active_alert_class = -1;
static int active_alert_confidence = 0;
static int active_alert_release_misses = 0;

typedef enum
{
    ALERT_DECISION_BACKGROUND = 0,
    ALERT_DECISION_CANDIDATE,
    ALERT_DECISION_TRIGGERED,
    ALERT_DECISION_ACTIVE,
    ALERT_DECISION_COOLDOWN
} alert_decision_kind_t;

typedef struct
{
    alert_decision_kind_t kind;
    int hit_count;
    int required_hits;
} alert_decision_t;

static void reset_alert_hit_counts(void)
{
    for (int i = 0; i < ALERT_CLASS_COUNT; i++)
    {
        alert_hit_count[i] = 0;
    }
}

static alert_decision_t handle_alert_logic(int predicted_class, int confidence_percent)
{
    alert_decision_t decision = {ALERT_DECISION_BACKGROUND, 0, 0};

    /*
     * Once an alert is verified, keep it active while the same reliable
     * sound continues. A short one-frame model wobble is tolerated so the
     * LCD does not fall back to VERIFYING in the middle of one real event.
     */
    if (active_alert_class >= 0)
    {
        int same_reliable_sound =
            (predicted_class == active_alert_class) &&
            (confidence_percent >= alert_threshold_percent[active_alert_class]);

        if (same_reliable_sound)
        {
            active_alert_confidence = confidence_percent;
            active_alert_release_misses = 0;
            decision.kind = ALERT_DECISION_ACTIVE;

            rt_kprintf("ALERT ACTIVE: %s continues at %d%%. Verification bypassed.\n",
                       alert_class_names[active_alert_class],
                       confidence_percent);
            return decision;
        }

        active_alert_release_misses++;

        if (active_alert_release_misses < ALERT_RELEASE_MISSES)
        {
            decision.kind = ALERT_DECISION_ACTIVE;
            rt_kprintf("ALERT ACTIVE: Holding %s through release check %d/%d.\n",
                       alert_class_names[active_alert_class],
                       active_alert_release_misses,
                       ALERT_RELEASE_MISSES);
            return decision;
        }

        rt_kprintf("ALERT CLEARED: %s is no longer continuously detected.\n",
                   alert_class_names[active_alert_class]);
        active_alert_class = -1;
        active_alert_confidence = 0;
        active_alert_release_misses = 0;
        reset_alert_hit_counts();
    }

    if ((predicted_class < 0) || (predicted_class >= ALERT_CLASS_COUNT))
    {
        rt_kprintf("ALERT STATUS: Invalid prediction. No alert.\n");
        reset_alert_hit_counts();
        return decision;
    }

    const char *label = alert_class_names[predicted_class];

    if (predicted_class == UNKNOWN_CLASS_INDEX)
    {
        rt_kprintf("ALERT STATUS: Unknown/Background detected. No alert.\n");
        reset_alert_hit_counts();
        return decision;
    }

    int threshold = alert_threshold_percent[predicted_class];

    if (confidence_percent < threshold)
    {
        rt_kprintf("ALERT STATUS: No reliable alert. %s confidence %d%% < threshold %d%%.\n",
                   label,
                   confidence_percent,
                   threshold);
        reset_alert_hit_counts();
        return decision;
    }

    /* Consecutive hit logic: reset other class counters. */
    for (int i = 0; i < ALERT_CLASS_COUNT; i++)
    {
        if (i != predicted_class)
        {
            alert_hit_count[i] = 0;
        }
    }

    alert_hit_count[predicted_class]++;
    decision.hit_count = alert_hit_count[predicted_class];
    decision.required_hits = alert_required_hits[predicted_class];

    rt_kprintf("ALERT CANDIDATE: %s confidence %d%%, hit %d/%d.\n",
               label,
               confidence_percent,
               alert_hit_count[predicted_class],
               alert_required_hits[predicted_class]);

    if (alert_hit_count[predicted_class] < alert_required_hits[predicted_class])
    {
        rt_kprintf("ALERT STATUS: Waiting for next confirmation.\n");
        decision.kind = ALERT_DECISION_CANDIDATE;
        return decision;
    }

    rt_tick_t now = rt_tick_get();
    rt_tick_t cooldown_ticks = rt_tick_from_millisecond(ALERT_COOLDOWN_MS);

    if ((alert_last_tick[predicted_class] != 0) &&
        ((now - alert_last_tick[predicted_class]) < cooldown_ticks))
    {
        rt_kprintf("ALERT STATUS: %s detected but suppressed by cooldown.\n", label);
        alert_hit_count[predicted_class] = 0;
        decision.kind = ALERT_DECISION_COOLDOWN;
        return decision;
    }

    rt_kprintf("\n********************************************\n");
    rt_kprintf("ALERT SOUND DETECTED: %s\n", label);
    rt_kprintf("Confidence: %d%%\n", confidence_percent);
    rt_kprintf("********************************************\n\n");

    alert_last_tick[predicted_class] = now;
    alert_hit_count[predicted_class] = 0;
    active_alert_class = predicted_class;
    active_alert_confidence = confidence_percent;
    active_alert_release_misses = 0;
    decision.kind = ALERT_DECISION_TRIGGERED;
    return decision;
}

static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

static uint16_t audio_buffer[AUDIO_SAMPLE_COUNT];
static int16_t audio_centered[AUDIO_SAMPLE_COUNT];

/* Final demo signal information */
static int demo_last_p2p = 0;
static int demo_last_sample_rate = 0;


/*
 * Audio after pre-emphasis filter.
 * This matches the Python training preprocessing.
 */
static float audio_preemphasized[AUDIO_SAMPLE_COUNT];

static uint32_t frame_energy[FRAME_COUNT];

static int8_t feature_buffer[FEATURE_LEN];

static float hamming_window[FRAME_SIZE];
static float windowed_frame[FRAME_SIZE];

static arm_rfft_fast_instance_f32 rfft_instance;

static float fft_input[FFT_SIZE];
static float fft_output[FFT_SIZE];
static float fft_power[FFT_BIN_COUNT];
static float fft_magnitude[FFT_BIN_COUNT];

static float mel_freq_hz[MEL_FILTER_COUNT + 2];
static float mel_energies[MEL_FILTER_COUNT];
static float log_mel_energies[MEL_FILTER_COUNT];

static float dct_cos_table[MFCC_COEFF_COUNT][MEL_FILTER_COUNT];
static float mfcc_coeffs[MFCC_COEFF_COUNT];

static float mfcc_features[MFCC_COEFF_COUNT][FRAME_COUNT];

static float delta_features[MFCC_COEFF_COUNT][FRAME_COUNT];
static float delta_delta_features[MFCC_COEFF_COUNT][FRAME_COUNT];

extern int cpp_tflm_init(const unsigned char *model_data,
                         unsigned int model_len,
                         unsigned char *arena,
                         unsigned int arena_size);

extern int cpp_tflm_run_features(const int8_t *features,
                                 unsigned int feature_len,
                                 int *predicted_class,
                                 int *confidence_percent);

static int mic_adc_init(void)
{
    fsp_err_t err;

    rt_kprintf("Opening ADC0...\n");

    err = R_ADC_B_Open(&g_adc0_ctrl, &g_adc0_cfg);
    if (err != FSP_SUCCESS)
    {
        rt_kprintf("ADC open failed: %d\n", err);
        return -1;
    }

    err = R_ADC_B_ScanCfg(&g_adc0_ctrl, &g_adc0_scan_cfg);
    if (err != FSP_SUCCESS)
    {
        rt_kprintf("ADC scan config failed: %d\n", err);
        return -2;
    }

    rt_kprintf("ADC0 ready.\n");

    return 0;
}

static uint16_t mic_adc_read_once(void)
{
    uint16_t adc_raw = 0;

    R_ADC_B_ScanStart(&g_adc0_ctrl);

    R_BSP_SoftwareDelay(ADC_SETTLE_US, BSP_DELAY_UNITS_MICROSECONDS);

    R_ADC_B_Read(&g_adc0_ctrl, MIC_ADC_CHANNEL, &adc_raw);

    return adc_raw;
}

static void capture_audio_buffer(void)
{
    rt_tick_t start_tick;
    rt_tick_t end_tick;

    for (int i = 0; i < 100; i++)
    {
        mic_adc_read_once();
        R_BSP_SoftwareDelay(SAMPLE_PERIOD_US, BSP_DELAY_UNITS_MICROSECONDS);
    }

    start_tick = rt_tick_get();

    for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
    {
        audio_buffer[i] = mic_adc_read_once();
        R_BSP_SoftwareDelay(SAMPLE_PERIOD_US, BSP_DELAY_UNITS_MICROSECONDS);
    }

    end_tick = rt_tick_get();

    int elapsed_ms = (int)(((end_tick - start_tick) * 1000) / RT_TICK_PER_SECOND);

    rt_kprintf("Captured %d ADC samples.\n", AUDIO_SAMPLE_COUNT);
    rt_kprintf("Approx capture time: %d ms\n", elapsed_ms);

    if (elapsed_ms > 0)
    {
        int approx_sample_rate = (AUDIO_SAMPLE_COUNT * 1000) / elapsed_ms;
        rt_kprintf("Approx sample rate: %d Hz\n", approx_sample_rate);
    }
}

static uint32_t analyze_raw_audio_buffer(void)
{
    uint16_t min_val = 65535;
    uint16_t max_val = 0;
    uint32_t sum = 0;

    for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
    {
        uint16_t value = audio_buffer[i];

        if (value < min_val)
        {
            min_val = value;
        }

        if (value > max_val)
        {
            max_val = value;
        }

        sum += value;
    }

    uint32_t avg_val = sum / AUDIO_SAMPLE_COUNT;
    uint16_t p2p_val = max_val - min_val;

    rt_kprintf("Raw ADC audio stats:\n");
    rt_kprintf("  min = %d\n", (int)min_val);
    rt_kprintf("  max = %d\n", (int)max_val);
    rt_kprintf("  avg = %d\n", (int)avg_val);
    rt_kprintf("  p2p = %d\n", (int)p2p_val);

    return avg_val;
}

static void convert_to_centered_audio(uint32_t dc_offset)
{
    for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
    {
        int32_t centered = (int32_t)audio_buffer[i] - (int32_t)dc_offset;

        if (centered > 32767)
        {
            centered = 32767;
        }

        if (centered < -32768)
        {
            centered = -32768;
        }

        audio_centered[i] = (int16_t)centered;
    }
}

static void apply_pre_emphasis(void)
{
    /*
     * Convert centered ADC samples to audio-like float range.
     * Training audio from librosa is in approximately -1.0 to +1.0.
     * ADC centered values are still in raw counts, so scale by 32768.
     */
    float current_sample = (float)audio_centered[0] / 32768.0f;

    audio_preemphasized[0] = current_sample * 0.03f;

    for (int i = 1; i < AUDIO_SAMPLE_COUNT; i++)
    {
        float current = (float)audio_centered[i] / 32768.0f;
        float previous = (float)audio_centered[i - 1] / 32768.0f;

        audio_preemphasized[i] = current - (0.97f * previous);
    }
}

static void analyze_centered_audio(void)
{
    int16_t min_val = 32767;
    int16_t max_val = -32768;
    int64_t sum = 0;

    for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
    {
        int16_t value = audio_centered[i];

        if (value < min_val)
        {
            min_val = value;
        }

        if (value > max_val)
        {
            max_val = value;
        }

        sum += value;
    }

    int32_t avg_val = (int32_t)(sum / AUDIO_SAMPLE_COUNT);
    int32_t p2p_val = (int32_t)max_val - (int32_t)min_val;

    rt_kprintf("Centered audio stats:\n");
    rt_kprintf("  min = %d\n", (int)min_val);
    rt_kprintf("  max = %d\n", (int)max_val);
    rt_kprintf("  avg = %d\n", (int)avg_val);
    rt_kprintf("  p2p = %d\n", (int)p2p_val);
}

static void calculate_frame_energy(void)
{
    uint32_t min_energy = 0xFFFFFFFF;
    uint32_t max_energy = 0;
    uint32_t sum_energy = 0;
    int max_energy_frame = 0;

    for (int frame = 0; frame < FRAME_COUNT; frame++)
    {
        int start_index = frame * FRAME_STEP;
        uint32_t sum_abs = 0;

        for (int j = 0; j < FRAME_SIZE; j++)
        {
            int index = start_index + j;
            float sample = audio_preemphasized[index] * hamming_window[j];

            if (sample < 0.0f)
            {
                sample = -sample;
            }

            sum_abs += (uint32_t)sample;
        }

        frame_energy[frame] = sum_abs / FRAME_SIZE;

        if (frame_energy[frame] < min_energy)
        {
            min_energy = frame_energy[frame];
        }

        if (frame_energy[frame] > max_energy)
        {
            max_energy = frame_energy[frame];
            max_energy_frame = frame;
        }

        sum_energy += frame_energy[frame];
    }

    uint32_t avg_energy = sum_energy / FRAME_COUNT;

    rt_kprintf("Frame energy summary:\n");
    rt_kprintf("  min energy = %d\n", (int)min_energy);
    rt_kprintf("  max energy = %d\n", (int)max_energy);
    rt_kprintf("  avg energy = %d\n", (int)avg_energy);
    rt_kprintf("  loudest frame = %d\n", max_energy_frame);

    rt_kprintf("First 10 frame energies:\n");

    for (int i = 0; i < 10; i++)
    {
        rt_kprintf("%d ", (int)frame_energy[i]);
    }

    rt_kprintf("\n");
}

static int clamp_int8(int value)
{
    if (value > 127)
    {
        return 127;
    }

    if (value < -128)
    {
        return -128;
    }

    return value;
}

static void fill_feature_buffer_from_frame_energy(void)
{
    int min_feature = 127;
    int max_feature = -128;
    int32_t sum_feature = 0;

    /*
     * Input tensor shape is 39 x 61.
     * Treat it as:
     * 39 pseudo coefficients
     * 61 time frames
     *
     * Flat index for [coeff][frame]:
     * index = coeff * 61 + frame
     */
    for (int frame = 0; frame < FEATURE_FRAMES; frame++)
    {
        /*
         * Scale energy into useful int8 range.
         * Quiet energy around 400 to 800 becomes small.
         * Loud energy around 6000 to 8000 becomes large.
         */
        int energy_scaled = (int)frame_energy[frame] / 80;

        if (energy_scaled > 100)
        {
            energy_scaled = 100;
        }

        for (int coeff = 0; coeff < FEATURE_COEFFS; coeff++)
        {
            int feature_value;

            /*
             * Pseudo MFCC-like pattern:
             * lower coefficients get stronger positive values,
             * middle coefficients get smaller values,
             * higher coefficients can become slightly negative.
             *
             * This is NOT real MFCC yet.
             * It only allows live audio energy to affect the model input.
             */
            if (coeff < 13)
            {
                feature_value = -5 + energy_scaled;
            }
            else if (coeff < 26)
            {
                feature_value = -5 + (energy_scaled / 2);
            }
            else
            {
                feature_value = -5 - (energy_scaled / 3);
            }

            /*
             * Add small variation across coefficients.
             */
            feature_value += ((coeff % 5) - 2) * energy_scaled / 10;

            feature_value = clamp_int8(feature_value);

            int index = (coeff * FEATURE_FRAMES) + frame;

            feature_buffer[index] = (int8_t)feature_value;

            if (feature_value < min_feature)
            {
                min_feature = feature_value;
            }

            if (feature_value > max_feature)
            {
                max_feature = feature_value;
            }

            sum_feature += feature_value;
        }
    }

    int avg_feature = sum_feature / FEATURE_LEN;

    rt_kprintf("Pseudo feature buffer summary:\n");
    rt_kprintf("  feature length = %d\n", FEATURE_LEN);
    rt_kprintf("  min feature = %d\n", min_feature);
    rt_kprintf("  max feature = %d\n", max_feature);
    rt_kprintf("  avg feature = %d\n", avg_feature);

    rt_kprintf("First 20 feature values:\n");

    for (int i = 0; i < 20; i++)
    {
        rt_kprintf("%d ", (int)feature_buffer[i]);
    }

    rt_kprintf("\n");
}

static void init_hamming_window(void)
{
    for (int i = 0; i < FRAME_SIZE; i++)
    {
        hamming_window[i] =
            0.54f - 0.46f * cosf((2.0f * 3.14159265f * i) / (FRAME_SIZE - 1));
    }

    rt_kprintf("Hamming window initialized.\n");
    rt_kprintf("hamming[0] x1000 = %d\n", (int)(hamming_window[0] * 1000.0f));
    rt_kprintf("hamming[256] x1000 = %d\n", (int)(hamming_window[256] * 1000.0f));
    rt_kprintf("hamming[511] x1000 = %d\n", (int)(hamming_window[511] * 1000.0f));
}

/* static void prepare_windowed_frame(int frame_index)
{
    int start_index = frame_index * FRAME_STEP;

    float min_val = 999999.0f;
    float max_val = -999999.0f;
    float sum_abs = 0.0f;

    for (int j = 0; j < FRAME_SIZE; j++)
    {
        int index = start_index + j;

        windowed_frame[j] = audio_preemphasized[index] * hamming_window[j];

        if (windowed_frame[j] < min_val)
        {
            min_val = windowed_frame[j];
        }

        if (windowed_frame[j] > max_val)
        {
            max_val = windowed_frame[j];
        }

        float abs_val = windowed_frame[j];

        if (abs_val < 0.0f)
        {
            abs_val = -abs_val;
        }

        sum_abs += abs_val;
    }

    rt_kprintf("Windowed frame test:\n");
    rt_kprintf("  frame index = %d\n", frame_index);
    rt_kprintf("  min x1000 = %d\n", (int)(min_val * 1000.0f));
    rt_kprintf("  max x1000 = %d\n", (int)(max_val * 1000.0f));
    rt_kprintf("  avg abs x1000 = %d\n", (int)((sum_abs / FRAME_SIZE) * 1000.0f));

    rt_kprintf("First 10 windowed samples x1000:\n");
    for (int i = 0; i < 10; i++)
    {
        rt_kprintf("%d ", (int)(windowed_frame[i] * 1000.0f));
    }
    rt_kprintf("\n");
} */

static int init_fft(void)
{
    arm_status status;

    status = arm_rfft_fast_init_f32(&rfft_instance, FFT_SIZE);

    if (status != ARM_MATH_SUCCESS)
    {
        rt_kprintf("FFT init failed: %d\n", (int)status);
        return -1;
    }

    rt_kprintf("FFT initialized.\n");
    rt_kprintf("FFT size: %d\n", FFT_SIZE);
    rt_kprintf("FFT bin count: %d\n", FFT_BIN_COUNT);

    return 0;
}

static void run_fft_test_on_frame(int frame_index)
{
    int start_index = frame_index * FRAME_STEP;

    for (int j = 0; j < FFT_SIZE; j++)
    {
        int index = start_index + j;

        windowed_frame[j] = audio_preemphasized[index] * hamming_window[j];
        fft_input[j] = windowed_frame[j];
    }

    arm_rfft_fast_f32(&rfft_instance, fft_input, fft_output, 0);

    fft_power[0] = fft_output[0] * fft_output[0];
    fft_magnitude[0] = fabsf(fft_output[0]);

    fft_power[FFT_SIZE / 2] = fft_output[1] * fft_output[1];
    fft_magnitude[FFT_SIZE / 2] = fabsf(fft_output[1]);

    for (int bin = 1; bin < (FFT_SIZE / 2); bin++)
    {
        float real = fft_output[2 * bin];
        float imag = fft_output[(2 * bin) + 1];

        fft_power[bin] = (real * real) + (imag * imag);
        fft_magnitude[bin] = sqrtf(fft_power[bin]);
    }

    float max_power = 0.0f;
    int max_power_bin = 0;

    for (int bin = 0; bin < FFT_BIN_COUNT; bin++)
    {
        if (fft_power[bin] > max_power)
        {
            max_power = fft_power[bin];
            max_power_bin = bin;
        }
    }

    rt_kprintf("FFT power summary:\n");
    rt_kprintf("  max power bin = %d\n", max_power_bin);
    rt_kprintf("  max power x1000000 = %d\n", (int)(max_power * 1000000.0f));
    rt_kprintf("  power[0] x1000000 = %d\n", (int)(fft_power[0] * 1000000.0f));
    rt_kprintf("  power[10] x1000000 = %d\n", (int)(fft_power[10] * 1000000.0f));
    rt_kprintf("  power[50] x1000000 = %d\n", (int)(fft_power[50] * 1000000.0f));
    rt_kprintf("  power[100] x1000000 = %d\n", (int)(fft_power[100] * 1000000.0f));

    rt_kprintf("FFT test:\n");
    rt_kprintf("  frame index = %d\n", frame_index);

    rt_kprintf("  mag[0] x1000 = %d\n", (int)(fft_magnitude[0] * 1000.0f));
    rt_kprintf("  mag[1] x1000 = %d\n", (int)(fft_magnitude[1] * 1000.0f));
    rt_kprintf("  mag[2] x1000 = %d\n", (int)(fft_magnitude[2] * 1000.0f));
    rt_kprintf("  mag[10] x1000 = %d\n", (int)(fft_magnitude[10] * 1000.0f));
    rt_kprintf("  mag[50] x1000 = %d\n", (int)(fft_magnitude[50] * 1000.0f));
    rt_kprintf("  mag[100] x1000 = %d\n", (int)(fft_magnitude[100] * 1000.0f));
    rt_kprintf("  mag[256] x1000 = %d\n", (int)(fft_magnitude[256] * 1000.0f));

    rt_kprintf("First 10 FFT magnitudes x1000:\n");
    for (int i = 0; i < 10; i++)
    {
        rt_kprintf("%d ", (int)(fft_magnitude[i] * 1000.0f));
    }

    rt_kprintf("\n");
}

static float hz_to_mel_slaney(float hz)
{
    /*
     * Librosa default uses Slaney-style Mel scale when htk=False.
     */
    const float f_sp = 200.0f / 3.0f;

    if (hz < 1000.0f)
    {
        return hz / f_sp;
    }
    else
    {
        const float min_log_hz = 1000.0f;
        const float min_log_mel = 1000.0f / f_sp;
        const float logstep = 0.068751777f; /* log(6.4) / 27 */

        return min_log_mel + (logf(hz / min_log_hz) / logstep);
    }
}

static float mel_to_hz_slaney(float mel)
{
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = 1000.0f / f_sp;
    const float logstep = 0.068751777f;

    if (mel < min_log_mel)
    {
        return mel * f_sp;
    }
    else
    {
        return min_log_hz * expf(logstep * (mel - min_log_mel));
    }
}

static void init_mel_filterbank(void)
{
    float mel_min = hz_to_mel_slaney(MEL_FMIN_HZ);
    float mel_max = hz_to_mel_slaney(MEL_FMAX_HZ);

    for (int i = 0; i < (MEL_FILTER_COUNT + 2); i++)
    {
        float mel_value =
            mel_min +
            ((mel_max - mel_min) * (float)i / (float)(MEL_FILTER_COUNT + 1));

        mel_freq_hz[i] = mel_to_hz_slaney(mel_value);
    }

    rt_kprintf("Mel filterbank initialized.\n");
    rt_kprintf("Mel filters: %d\n", MEL_FILTER_COUNT);
    rt_kprintf("mel_freq_hz[0] x1000 = %d\n", (int)(mel_freq_hz[0] * 1000.0f));
    rt_kprintf("mel_freq_hz[1] x1000 = %d\n", (int)(mel_freq_hz[1] * 1000.0f));
    rt_kprintf("mel_freq_hz[20] x1000 = %d\n", (int)(mel_freq_hz[20] * 1000.0f));
    rt_kprintf("mel_freq_hz[41] x1000 = %d\n", (int)(mel_freq_hz[41] * 1000.0f));
}

static void calculate_mel_energies_from_fft_power(void)
{
    for (int mel = 0; mel < MEL_FILTER_COUNT; mel++)
    {
        float left_hz = mel_freq_hz[mel];
        float center_hz = mel_freq_hz[mel + 1];
        float right_hz = mel_freq_hz[mel + 2];

        float energy = 0.0f;

        for (int bin = 0; bin < FFT_BIN_COUNT; bin++)
        {
            float bin_hz =
                ((float)TARGET_SAMPLE_RATE_HZ * (float)bin) / (float)FFT_SIZE;

            float weight = 0.0f;

            if ((bin_hz >= left_hz) && (bin_hz <= center_hz))
            {
                weight = (bin_hz - left_hz) / (center_hz - left_hz);
            }
            else if ((bin_hz > center_hz) && (bin_hz <= right_hz))
            {
                weight = (right_hz - bin_hz) / (right_hz - center_hz);
            }

            /*
             * Slaney-style normalization, similar to librosa norm='slaney'.
             */
            weight = weight * (2.0f / (right_hz - left_hz));

            energy += fft_power[bin] * weight;
        }

        if (energy < 1.0e-10f)
        {
            energy = 1.0e-10f;
        }

        mel_energies[mel] = energy;

        /*
         * Librosa MFCC uses power-to-dB before DCT.
         * This is 10 * log10(power).
         */
        log_mel_energies[mel] = 10.0f * log10f(energy);
    }

    float max_mel = 0.0f;
    int max_mel_index = 0;

    for (int mel = 0; mel < MEL_FILTER_COUNT; mel++)
    {
        if (mel_energies[mel] > max_mel)
        {
            max_mel = mel_energies[mel];
            max_mel_index = mel;
        }
    }

    rt_kprintf("Mel energy summary:\n");
    rt_kprintf("  max mel index = %d\n", max_mel_index);
    rt_kprintf("  max mel /1e6 = %d\n", (int)(max_mel / 1000000.0f));

    rt_kprintf("  mel[0] /1e6 = %d\n", (int)(mel_energies[0] / 1000000.0f));
    rt_kprintf("  mel[10] /1e6 = %d\n", (int)(mel_energies[10] / 1000000.0f));
    rt_kprintf("  mel[20] /1e6 = %d\n", (int)(mel_energies[20] / 1000000.0f));
    rt_kprintf("  mel[39] /1e6 = %d\n", (int)(mel_energies[39] / 1000000.0f));

    rt_kprintf("Log Mel values x1000:\n");
    rt_kprintf("  logmel[0] = %d\n", (int)(log_mel_energies[0] * 1000.0f));
    rt_kprintf("  logmel[10] = %d\n", (int)(log_mel_energies[10] * 1000.0f));
    rt_kprintf("  logmel[20] = %d\n", (int)(log_mel_energies[20] * 1000.0f));
    rt_kprintf("  logmel[39] = %d\n", (int)(log_mel_energies[39] * 1000.0f));
}

static void init_dct_table(void)
{
    for (int k = 0; k < MFCC_COEFF_COUNT; k++)
    {
        for (int n = 0; n < MEL_FILTER_COUNT; n++)
        {
            dct_cos_table[k][n] =
                cosf((3.14159265f * (float)k * ((2.0f * (float)n) + 1.0f)) /
                     (2.0f * (float)MEL_FILTER_COUNT));
        }
    }

    rt_kprintf("DCT table initialized.\n");
    rt_kprintf("MFCC coeff count: %d\n", MFCC_COEFF_COUNT);
    rt_kprintf("dct_cos[0][0] x1000 = %d\n", (int)(dct_cos_table[0][0] * 1000.0f));
    rt_kprintf("dct_cos[1][0] x1000 = %d\n", (int)(dct_cos_table[1][0] * 1000.0f));
    rt_kprintf("dct_cos[12][39] x1000 = %d\n", (int)(dct_cos_table[12][39] * 1000.0f));
}

static void calculate_mfcc_from_log_mel(void)
{
    /*
     * This follows DCT type-II with orthonormal scaling,
     * similar to librosa.feature.mfcc default.
     */
    for (int k = 0; k < MFCC_COEFF_COUNT; k++)
    {
        float sum = 0.0f;

        for (int n = 0; n < MEL_FILTER_COUNT; n++)
        {
            sum += log_mel_energies[n] * dct_cos_table[k][n];
        }

        if (k == 0)
        {
            mfcc_coeffs[k] = sum * sqrtf(1.0f / (float)MEL_FILTER_COUNT);
        }
        else
        {
            mfcc_coeffs[k] = sum * sqrtf(2.0f / (float)MEL_FILTER_COUNT);
        }
    }

    rt_kprintf("MFCC test for frame 0:\n");

    for (int i = 0; i < MFCC_COEFF_COUNT; i++)
    {
        rt_kprintf("  mfcc[%d] x1000 = %d\n",
                   i,
                   (int)(mfcc_coeffs[i] * 1000.0f));
    }
}

static void calculate_fft_power_for_frame(int frame_index)
{
    int start_index = frame_index * FRAME_STEP;

    for (int j = 0; j < FFT_SIZE; j++)
    {
        int index = start_index + j;

        windowed_frame[j] = audio_preemphasized[index] * hamming_window[j];
        fft_input[j] = windowed_frame[j];
    }

    arm_rfft_fast_f32(&rfft_instance, fft_input, fft_output, 0);

    fft_power[0] = fft_output[0] * fft_output[0];
    fft_magnitude[0] = fabsf(fft_output[0]);

    fft_power[FFT_SIZE / 2] = fft_output[1] * fft_output[1];
    fft_magnitude[FFT_SIZE / 2] = fabsf(fft_output[1]);

    for (int bin = 1; bin < (FFT_SIZE / 2); bin++)
    {
        float real = fft_output[2 * bin];
        float imag = fft_output[(2 * bin) + 1];

        fft_power[bin] = (real * real) + (imag * imag);
        fft_magnitude[bin] = sqrtf(fft_power[bin]);
    }
}

static void calculate_mel_energies_silent(void)
{
    for (int mel = 0; mel < MEL_FILTER_COUNT; mel++)
    {
        float left_hz = mel_freq_hz[mel];
        float center_hz = mel_freq_hz[mel + 1];
        float right_hz = mel_freq_hz[mel + 2];

        float energy = 0.0f;

        for (int bin = 0; bin < FFT_BIN_COUNT; bin++)
        {
            float bin_hz =
                ((float)TARGET_SAMPLE_RATE_HZ * (float)bin) / (float)FFT_SIZE;

            float weight = 0.0f;

            if ((bin_hz >= left_hz) && (bin_hz <= center_hz))
            {
                weight = (bin_hz - left_hz) / (center_hz - left_hz);
            }
            else if ((bin_hz > center_hz) && (bin_hz <= right_hz))
            {
                weight = (right_hz - bin_hz) / (right_hz - center_hz);
            }

            weight = weight * (2.0f / (right_hz - left_hz));

            energy += fft_power[bin] * weight;
        }

        if (energy < 1.0e-10f)
        {
            energy = 1.0e-10f;
        }

        mel_energies[mel] = energy;
        log_mel_energies[mel] = 10.0f * log10f(energy);
    }
}

static void calculate_mfcc_for_current_frame(int frame_index)
{
    for (int k = 0; k < MFCC_COEFF_COUNT; k++)
    {
        float sum = 0.0f;

        for (int n = 0; n < MEL_FILTER_COUNT; n++)
        {
            sum += log_mel_energies[n] * dct_cos_table[k][n];
        }

        if (k == 0)
        {
            mfcc_features[k][frame_index] =
                sum * sqrtf(1.0f / (float)MEL_FILTER_COUNT);
        }
        else
        {
            mfcc_features[k][frame_index] =
                sum * sqrtf(2.0f / (float)MEL_FILTER_COUNT);
        }
    }
}

static void calculate_all_mfcc_frames(void)
{
    for (int frame = 0; frame < FRAME_COUNT; frame++)
    {
        calculate_fft_power_for_frame(frame);
        calculate_mel_energies_silent();
        calculate_mfcc_for_current_frame(frame);
    }

    float min_mfcc = 999999.0f;
    float max_mfcc = -999999.0f;
    float sum_abs = 0.0f;

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float value = mfcc_features[coeff][frame];

            if (value < min_mfcc)
            {
                min_mfcc = value;
            }

            if (value > max_mfcc)
            {
                max_mfcc = value;
            }

            if (value < 0.0f)
            {
                sum_abs -= value;
            }
            else
            {
                sum_abs += value;
            }
        }
    }

    rt_kprintf("All MFCC frames calculated.\n");
    rt_kprintf("  shape = 13 x 61\n");
    rt_kprintf("  min mfcc x1000 = %d\n", (int)(min_mfcc * 1000.0f));
    rt_kprintf("  max mfcc x1000 = %d\n", (int)(max_mfcc * 1000.0f));
    rt_kprintf("  avg abs mfcc x1000 = %d\n",
               (int)((sum_abs / (MFCC_COEFF_COUNT * FRAME_COUNT)) * 1000.0f));

    rt_kprintf("First 13 MFCC values for frame 0 x1000:\n");

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        rt_kprintf("%d ", (int)(mfcc_features[coeff][0] * 1000.0f));
    }

    rt_kprintf("\n");
}

static int clamp_frame_index(int index)
{
    if (index < 0)
    {
        return 0;
    }

    if (index >= FRAME_COUNT)
    {
        return FRAME_COUNT - 1;
    }

    return index;
}

static void calculate_delta_features(float input[MFCC_COEFF_COUNT][FRAME_COUNT],
                                     float output[MFCC_COEFF_COUNT][FRAME_COUNT])
{
    /*
     * Delta approximation using N = 4.
     * This matches the common speech-processing delta formula:
     * delta[t] = sum(n * (x[t+n] - x[t-n])) / (2 * sum(n^2))
     */
    const int N = 4;
    const float denominator = 60.0f; /* 2 * (1^2 + 2^2 + 3^2 + 4^2) */

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float numerator = 0.0f;

            for (int n = 1; n <= N; n++)
            {
                int prev_index = clamp_frame_index(frame - n);
                int next_index = clamp_frame_index(frame + n);

                numerator +=
                    (float)n *
                    (input[coeff][next_index] - input[coeff][prev_index]);
            }

            output[coeff][frame] = numerator / denominator;
        }
    }
}

static void calculate_all_delta_features(void)
{
    calculate_delta_features(mfcc_features, delta_features);
    calculate_delta_features(delta_features, delta_delta_features);

    float min_delta = 999999.0f;
    float max_delta = -999999.0f;
    float min_delta2 = 999999.0f;
    float max_delta2 = -999999.0f;

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float d1 = delta_features[coeff][frame];
            float d2 = delta_delta_features[coeff][frame];

            if (d1 < min_delta) min_delta = d1;
            if (d1 > max_delta) max_delta = d1;

            if (d2 < min_delta2) min_delta2 = d2;
            if (d2 > max_delta2) max_delta2 = d2;
        }
    }

    rt_kprintf("Delta features calculated.\n");
    rt_kprintf("  delta shape = 13 x 61\n");
    rt_kprintf("  delta min x1000 = %d\n", (int)(min_delta * 1000.0f));
    rt_kprintf("  delta max x1000 = %d\n", (int)(max_delta * 1000.0f));

    rt_kprintf("Delta-delta features calculated.\n");
    rt_kprintf("  delta-delta shape = 13 x 61\n");
    rt_kprintf("  delta2 min x1000 = %d\n", (int)(min_delta2 * 1000.0f));
    rt_kprintf("  delta2 max x1000 = %d\n", (int)(max_delta2 * 1000.0f));

    rt_kprintf("First 13 delta values for frame 0 x1000:\n");
    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        rt_kprintf("%d ", (int)(delta_features[coeff][0] * 1000.0f));
    }
    rt_kprintf("\n");

    rt_kprintf("First 13 delta-delta values for frame 0 x1000:\n");
    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        rt_kprintf("%d ", (int)(delta_delta_features[coeff][0] * 1000.0f));
    }
    rt_kprintf("\n");
}

static int clamp_to_int8(int value)
{
    if (value > 127)
    {
        return 127;
    }

    if (value < -128)
    {
        return -128;
    }

    return value;
}

static void fill_feature_buffer_from_real_mfcc(void)
{
    int index = 0;

    int min_q = 127;
    int max_q = -128;
    int sum_q = 0;

    float min_norm = 999999.0f;
    float max_norm = -999999.0f;

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float raw_value = mfcc_features[coeff][frame];

            float normalized =
                (raw_value - mfcc_mean[index]) / mfcc_std[index];

            int q =
                (int)roundf((normalized / MODEL_INPUT_SCALE) +
                            MODEL_INPUT_ZERO_POINT);

            q = clamp_to_int8(q);

            feature_buffer[index] = (int8_t)q;

            if (q < min_q) min_q = q;
            if (q > max_q) max_q = q;
            sum_q += q;

            if (normalized < min_norm) min_norm = normalized;
            if (normalized > max_norm) max_norm = normalized;

            index++;
        }
    }

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float raw_value = delta_features[coeff][frame];

            float normalized =
                (raw_value - mfcc_mean[index]) / mfcc_std[index];

            int q =
                (int)roundf((normalized / MODEL_INPUT_SCALE) +
                            MODEL_INPUT_ZERO_POINT);

            q = clamp_to_int8(q);

            feature_buffer[index] = (int8_t)q;

            if (q < min_q) min_q = q;
            if (q > max_q) max_q = q;
            sum_q += q;

            if (normalized < min_norm) min_norm = normalized;
            if (normalized > max_norm) max_norm = normalized;

            index++;
        }
    }

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float raw_value = delta_delta_features[coeff][frame];

            float normalized =
                (raw_value - mfcc_mean[index]) / mfcc_std[index];

            int q =
                (int)roundf((normalized / MODEL_INPUT_SCALE) +
                            MODEL_INPUT_ZERO_POINT);

            q = clamp_to_int8(q);

            feature_buffer[index] = (int8_t)q;

            if (q < min_q) min_q = q;
            if (q > max_q) max_q = q;
            sum_q += q;

            if (normalized < min_norm) min_norm = normalized;
            if (normalized > max_norm) max_norm = normalized;

            index++;
        }
    }

    rt_kprintf("Real MFCC feature buffer summary:\n");
    rt_kprintf("  feature length = %d\n", index);
    rt_kprintf("  min q = %d\n", min_q);
    rt_kprintf("  max q = %d\n", max_q);
    rt_kprintf("  avg q = %d\n", sum_q / index);
    rt_kprintf("  min normalized x1000 = %d\n", (int)(min_norm * 1000.0f));
    rt_kprintf("  max normalized x1000 = %d\n", (int)(max_norm * 1000.0f));

    rt_kprintf("First 20 real feature values:\n");

    for (int i = 0; i < 20; i++)
    {
        rt_kprintf("%d ", feature_buffer[i]);
    }

    rt_kprintf("\n");
}


/* ============================================================
 * Final demo silent helper functions
 * ------------------------------------------------------------
 * These helpers avoid printing FFT/MFCC/debug values during demo.
 * The original debug functions are kept above for troubleshooting.
 * ============================================================ */
static void demo_capture_audio_buffer(void)
{
    rt_tick_t start_tick;
    rt_tick_t end_tick;

    for (int i = 0; i < 100; i++)
    {
        mic_adc_read_once();
        R_BSP_SoftwareDelay(SAMPLE_PERIOD_US, BSP_DELAY_UNITS_MICROSECONDS);
    }

    start_tick = rt_tick_get();

    for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
    {
        audio_buffer[i] = mic_adc_read_once();
        R_BSP_SoftwareDelay(SAMPLE_PERIOD_US, BSP_DELAY_UNITS_MICROSECONDS);
    }

    end_tick = rt_tick_get();

    int elapsed_ms = (int)(((end_tick - start_tick) * 1000) / RT_TICK_PER_SECOND);

    if (elapsed_ms > 0)
    {
        demo_last_sample_rate = (AUDIO_SAMPLE_COUNT * 1000) / elapsed_ms;
    }
    else
    {
        demo_last_sample_rate = 0;
    }
}

static uint32_t demo_analyze_raw_audio_buffer(void)
{
    uint16_t min_val = 65535;
    uint16_t max_val = 0;
    uint32_t sum = 0;

    for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
    {
        uint16_t value = audio_buffer[i];

        if (value < min_val)
        {
            min_val = value;
        }

        if (value > max_val)
        {
            max_val = value;
        }

        sum += value;
    }

    demo_last_p2p = (int)((uint32_t)max_val - (uint32_t)min_val);

    return sum / AUDIO_SAMPLE_COUNT;
}

static void demo_calculate_all_mfcc_frames(void)
{
    for (int frame = 0; frame < FRAME_COUNT; frame++)
    {
        calculate_fft_power_for_frame(frame);
        calculate_mel_energies_silent();
        calculate_mfcc_for_current_frame(frame);
    }
}

static void demo_calculate_all_delta_features(void)
{
    calculate_delta_features(mfcc_features, delta_features);
    calculate_delta_features(delta_features, delta_delta_features);
}

static void demo_fill_feature_buffer_from_real_mfcc(void)
{
    int index = 0;

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float raw_value = mfcc_features[coeff][frame];
            float normalized = (raw_value - mfcc_mean[index]) / mfcc_std[index];

            int q = (int)roundf((normalized / MODEL_INPUT_SCALE) +
                                MODEL_INPUT_ZERO_POINT);

            feature_buffer[index] = (int8_t)clamp_to_int8(q);
            index++;
        }
    }

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float raw_value = delta_features[coeff][frame];
            float normalized = (raw_value - mfcc_mean[index]) / mfcc_std[index];

            int q = (int)roundf((normalized / MODEL_INPUT_SCALE) +
                                MODEL_INPUT_ZERO_POINT);

            feature_buffer[index] = (int8_t)clamp_to_int8(q);
            index++;
        }
    }

    for (int coeff = 0; coeff < MFCC_COEFF_COUNT; coeff++)
    {
        for (int frame = 0; frame < FRAME_COUNT; frame++)
        {
            float raw_value = delta_delta_features[coeff][frame];
            float normalized = (raw_value - mfcc_mean[index]) / mfcc_std[index];

            int q = (int)roundf((normalized / MODEL_INPUT_SCALE) +
                                MODEL_INPUT_ZERO_POINT);

            feature_buffer[index] = (int8_t)clamp_to_int8(q);
            index++;
        }
    }
}


/* ============================================================
 * Audio dump collection mode function
 * ------------------------------------------------------------
 * This mode is only for dataset collection / thesis screenshot.
 * It captures one second of ADC audio, removes DC offset, and
 * prints centered samples between BEGIN_AUDIO and END_AUDIO.
 * ============================================================ */
static void audio_dump_collection_loop(void)
{
    int capture_count = 0;

    rt_kprintf("\n============================================\n");
    rt_kprintf("AUDIO DUMP COLLECTION MODE\n");
    rt_kprintf("============================================\n");
    rt_kprintf("This mode prints centered ADC samples for dataset collection.\n");
    rt_kprintf("Only use this mode for audio capture screenshots or WAV conversion.\n");
    rt_kprintf("Change AUDIO_DUMP_COLLECTION_MODE back to 0 for final demo mode.\n");

    while (1)
    {
        capture_count++;

        rt_kprintf("\n--------------------------------------------\n");
        rt_kprintf("Audio capture #%d will start in %d ms\n",
                   capture_count,
                   AUDIO_DUMP_START_DELAY_MS);
        rt_kprintf("Play the target sound now.\n");

        rt_thread_mdelay(AUDIO_DUMP_START_DELAY_MS);

        capture_audio_buffer();

        uint32_t dc_offset = analyze_raw_audio_buffer();
        rt_kprintf("Using DC offset = %d\n", (int)dc_offset);

        convert_to_centered_audio(dc_offset);
        analyze_centered_audio();

        rt_kprintf("BEGIN_AUDIO\n");
        for (int i = 0; i < AUDIO_SAMPLE_COUNT; i++)
        {
            rt_kprintf("%d\n", (int)audio_centered[i]);
        }
        rt_kprintf("END_AUDIO\n");

        rt_kprintf("Audio capture #%d complete. Waiting %d ms before next capture.\n",
                   capture_count,
                   AUDIO_DUMP_GAP_MS);
        rt_kprintf("--------------------------------------------\n");

        rt_thread_mdelay(AUDIO_DUMP_GAP_MS);
    }
}

static const char *demo_label_from_class(int predicted_class)
{
    if ((predicted_class < 0) || (predicted_class >= ALERT_CLASS_COUNT))
    {
        return "Invalid";
    }

    return alert_class_names[predicted_class];
}

static int demo_should_show_background(int predicted_class, int confidence_percent)
{
#if CLEAN_BACKGROUND_OUTPUT_MODE
    /*
     * Invalid output is treated as no reliable sound.
     */
    if ((predicted_class < 0) || (predicted_class >= ALERT_CLASS_COUNT))
    {
        return 1;
    }

    /*
     * If the raw model already predicts Unknown/Background,
     * display it as background.
     */
    if (predicted_class == UNKNOWN_CLASS_INDEX)
    {
        return 1;
    }

    /*
     * If signal level is very low, treat as background/no reliable sound.
     * This helps clean the demo when only room/background noise is present.
     */
    if (demo_last_p2p < DEMO_QUIET_P2P_THRESHOLD)
    {
        return 1;
    }

    /*
     * If the prediction is below the alert threshold,
     * hide the raw class and display it as Unknown/Background.
     *
     * Example:
     * Doorbell 60% but threshold is 75%
     * -> show Unknown/Background instead of Doorbell.
     */
    if (confidence_percent < alert_threshold_percent[predicted_class])
    {
        return 1;
    }
#else
    (void)predicted_class;
    (void)confidence_percent;
#endif

    return 0;
}

void hal_entry(void)
{
    volatile unsigned int checksum = 0;

    for (unsigned int i = 0; i < sound_model_len; i++)
    {
        checksum = checksum + sound_model[i];
    }

    rt_kprintf("\n============================================\n");
    rt_kprintf(" FINAL DEMO - Embedded Acoustic Event Detection\n");
    rt_kprintf("============================================\n");
    rt_kprintf("Model length: %d bytes\n", (int)sound_model_len);
    rt_kprintf("Model checksum: %u\n", checksum);
    rt_kprintf("Tensor arena: %d bytes\n", TENSOR_ARENA_SIZE);
    rt_kprintf("Audio frame: %d samples at target %d Hz\n", AUDIO_SAMPLE_COUNT, TARGET_SAMPLE_RATE_HZ);
    rt_kprintf("Feature input: %d x %d = %d int8 values\n", FEATURE_COEFFS, FEATURE_FRAMES, FEATURE_LEN);
    rt_kprintf("Alert thresholds: Baby 75%%, Glass 55%%, Alarm 75%%, Knock 80%%, Doorbell 75%%\n");
    rt_kprintf("Alert rule: repeated detections required except Glass Breaking.\n");
    rt_kprintf("Active alert rule: the same continuing sound stays latched until two release checks.\n");
    rt_kprintf("Unknown/Background does not trigger alert.\n");

    int lcd_ret = lcd_alert_ui_init();
    if (lcd_ret != RT_EOK)
    {
        rt_kprintf("LCD UI init failed: %d. Inference will continue on serial.\n", lcd_ret);
    }

    int adc_ret = mic_adc_init();
    if (adc_ret != 0)
    {
        rt_kprintf("ADC init failed: %d\n", adc_ret);
        lcd_alert_ui_show_error("ADC FAILURE");

        while (1)
        {
            rt_thread_mdelay(1000);
        }
    }

#if AUDIO_DUMP_COLLECTION_MODE
    audio_dump_collection_loop();
#endif

    int init_ret = cpp_tflm_init(sound_model,
                                 sound_model_len,
                                 tensor_arena,
                                 TENSOR_ARENA_SIZE);

    if (init_ret != 0)
    {
        rt_kprintf("TFLM init failed: %d\n", init_ret);
        lcd_alert_ui_show_error("MODEL FAILURE");

        while (1)
        {
            rt_thread_mdelay(1000);
        }
    }

    init_hamming_window();

    int fft_ret = init_fft();
    if (fft_ret != 0)
    {
        rt_kprintf("FFT init failed: %d\n", fft_ret);
        lcd_alert_ui_show_error("FFT FAILURE");

        while (1)
        {
            rt_thread_mdelay(1000);
        }
    }

    init_mel_filterbank();
    init_dct_table();

    rt_kprintf("\nSystem ready. Starting live inference...\n");
    lcd_alert_ui_show_monitoring();
    rt_kprintf("Serial console ready on UART8 at 115200 baud.\n");
    rt_kprintf("MSH command available: lcd_ui_test\n");

    /* Yield once so the RT-Thread shell can start before audio capture begins. */
    rt_thread_mdelay(500);

    int run_count = 0;

    while (1)
    {
        int predicted_class = -1;
        int confidence = 0;

        rt_kprintf("\n--------------------------------------------\n");
        rt_kprintf("Live run #%d\n", run_count + 1);

        demo_capture_audio_buffer();

        uint32_t dc_offset = demo_analyze_raw_audio_buffer();
        convert_to_centered_audio(dc_offset);
        apply_pre_emphasis();

        demo_calculate_all_mfcc_frames();
        demo_calculate_all_delta_features();
        demo_fill_feature_buffer_from_real_mfcc();

        rt_kprintf("Signal p2p: %d\n", demo_last_p2p);
        rt_kprintf("Sample rate approx: %d Hz\n", demo_last_sample_rate);

        int run_ret = cpp_tflm_run_features(feature_buffer,
                                            FEATURE_LEN,
                                            &predicted_class,
                                            &confidence);

        if (run_ret != 0)
        {
            rt_kprintf("Inference error: %d\n", run_ret);
            reset_alert_hit_counts();
            lcd_alert_ui_show_error("INFERENCE FAILURE");
        }
        else
        {
            int serial_display_class = predicted_class;
            alert_decision_t decision;

            if (demo_should_show_background(predicted_class, confidence))
            {
                serial_display_class = UNKNOWN_CLASS_INDEX;
                rt_kprintf("Prediction: Unknown/Background\n");
                decision = handle_alert_logic(UNKNOWN_CLASS_INDEX, confidence);
            }
            else
            {
                rt_kprintf("Prediction: %s\n", demo_label_from_class(predicted_class));
                rt_kprintf("Confidence: %d%%\n", confidence);

                decision = handle_alert_logic(predicted_class, confidence);
            }

            if ((decision.kind == ALERT_DECISION_TRIGGERED) ||
                (decision.kind == ALERT_DECISION_ACTIVE))
            {
                lcd_alert_ui_show_alert(active_alert_class, active_alert_confidence);
            }
            else if (decision.kind == ALERT_DECISION_CANDIDATE)
            {
                lcd_alert_ui_show_candidate(predicted_class,
                                            confidence,
                                            decision.hit_count,
                                            decision.required_hits);
            }
            else
            {
                lcd_alert_ui_show_monitoring();
            }

            rt_kprintf("[SERIAL/LCD] Run %d | LCD: %s | Sound: %s | Confidence: %d%% | P2P: %d\n",
                       run_count + 1,
                       lcd_alert_ui_state_name(),
                       demo_label_from_class(serial_display_class),
                       confidence,
                       demo_last_p2p);
        }

        rt_kprintf("--------------------------------------------\n");

        run_count++;
        rt_thread_mdelay(FINAL_DEMO_DELAY_MS);
    }
}
