extern "C" {
#include <rtthread.h>
#include <stdint.h>
}

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

static const int CLASS_LABEL_COUNT = 6;

static const char *class_labels[CLASS_LABEL_COUNT] =
{
    "Baby Crying",
    "Glass Breaking",
    "Alarm/Siren",
    "Door Knock",
    "Doorbell",
    "Unknown/Background"
};

static const char *get_class_label(int class_index)
{
    if ((class_index >= 0) && (class_index < CLASS_LABEL_COUNT))
    {
        return class_labels[class_index];
    }

    return "Unknown Index";
}

static tflite::MicroInterpreter *g_interpreter = nullptr;
static TfLiteTensor *g_input = nullptr;
static TfLiteTensor *g_output = nullptr;
static bool g_tflm_ready = false;

static const char *tensor_type_name(TfLiteType type)
{
    switch (type)
    {
        case kTfLiteFloat32: return "float32";
        case kTfLiteInt8: return "int8";
        case kTfLiteUInt8: return "uint8";
        case kTfLiteInt16: return "int16";
        case kTfLiteInt32: return "int32";
        default: return "unknown";
    }
}

static void print_tensor_info(const char *name, TfLiteTensor *tensor)
{
    if (tensor == nullptr)
    {
        rt_kprintf("%s tensor is NULL\n", name);
        return;
    }

    rt_kprintf("%s tensor:\n", name);
    rt_kprintf("  type: %s (%d)\n", tensor_type_name(tensor->type), (int)tensor->type);
    rt_kprintf("  bytes: %d\n", (int)tensor->bytes);

    if (tensor->dims != nullptr)
    {
        rt_kprintf("  dims size: %d\n", tensor->dims->size);
        rt_kprintf("  shape: ");

        for (int i = 0; i < tensor->dims->size; i++)
        {
            rt_kprintf("%d", tensor->dims->data[i]);

            if (i < tensor->dims->size - 1)
            {
                rt_kprintf(" x ");
            }
        }

        rt_kprintf("\n");
    }

    rt_kprintf("  quant scale: %d.%06d\n",
               (int)tensor->params.scale,
               (int)((tensor->params.scale - (int)tensor->params.scale) * 1000000));
    rt_kprintf("  quant zero_point: %d\n", (int)tensor->params.zero_point);
}

extern "C" int cpp_tflm_init(const unsigned char *model_data,
                             unsigned int model_len,
                             unsigned char *arena,
                             unsigned int arena_size)
{
    if (g_tflm_ready)
    {
        rt_kprintf("TFLM already initialized.\n");
        return 0;
    }

    rt_kprintf("C++ TFLM init started.\n");
    rt_kprintf("Model length: %d bytes\n", (int)model_len);
    rt_kprintf("Arena size: %d bytes\n", (int)arena_size);

    const tflite::Model *model = tflite::GetModel(model_data);

    if (model == nullptr)
    {
        rt_kprintf("TFLM ERROR: GetModel returned NULL.\n");
        return -1;
    }

    int model_version = model->version();

    rt_kprintf("Model schema version: %d\n", model_version);
    rt_kprintf("Expected TFLite schema version: %d\n", TFLITE_SCHEMA_VERSION);

    if (model_version != TFLITE_SCHEMA_VERSION)
    {
        rt_kprintf("TFLM ERROR: Schema mismatch.\n");
        return -2;
    }

    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter *error_reporter = &micro_error_reporter;

    static tflite::MicroMutableOpResolver<6> resolver;

    if (resolver.AddConv2D() != kTfLiteOk)
    {
        rt_kprintf("Failed to add CONV_2D\n");
        return -3;
    }

    if (resolver.AddMul() != kTfLiteOk)
    {
        rt_kprintf("Failed to add MUL\n");
        return -4;
    }

    if (resolver.AddAdd() != kTfLiteOk)
    {
        rt_kprintf("Failed to add ADD\n");
        return -5;
    }

    if (resolver.AddMaxPool2D() != kTfLiteOk)
    {
        rt_kprintf("Failed to add MAX_POOL_2D\n");
        return -6;
    }

    if (resolver.AddAveragePool2D() != kTfLiteOk)
    {
        rt_kprintf("Failed to add AVERAGE_POOL_2D\n");
        return -7;
    }

    if (resolver.AddSoftmax() != kTfLiteOk)
    {
        rt_kprintf("Failed to add SOFTMAX\n");
        return -8;
    }

    rt_kprintf("Resolver setup OK.\n");

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        arena,
        arena_size,
        error_reporter
    );

    g_interpreter = &interpreter;

    rt_kprintf("MicroInterpreter created.\n");
    rt_kprintf("Calling AllocateTensors...\n");

    TfLiteStatus allocate_status = g_interpreter->AllocateTensors();

    if (allocate_status != kTfLiteOk)
    {
        rt_kprintf("TFLM ERROR: AllocateTensors failed.\n");
        return -10;
    }

    rt_kprintf("AllocateTensors OK.\n");

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    print_tensor_info("Input", g_input);
    print_tensor_info("Output", g_output);

    g_tflm_ready = true;

    rt_kprintf("TFLM init complete.\n");

    return 0;
}

extern "C" int cpp_tflm_run_features(const int8_t *features,
                                     unsigned int feature_len,
                                     int *predicted_class,
                                     int *confidence_percent)
{
    if (!g_tflm_ready || g_interpreter == nullptr || g_input == nullptr || g_output == nullptr)
    {
        rt_kprintf("TFLM ERROR: Model not initialized.\n");
        return -1;
    }

    if (features == nullptr)
    {
        rt_kprintf("TFLM ERROR: Input feature pointer is NULL.\n");
        return -2;
    }

    if (feature_len != g_input->bytes)
    {
        rt_kprintf("TFLM ERROR: Feature length mismatch. Expected %d, got %d\n",
                   (int)g_input->bytes,
                   (int)feature_len);
        return -3;
    }

    for (unsigned int i = 0; i < g_input->bytes; i++)
    {
        g_input->data.int8[i] = features[i];
    }

    TfLiteStatus invoke_status = g_interpreter->Invoke();

    if (invoke_status != kTfLiteOk)
    {
        rt_kprintf("TFLM ERROR: Invoke failed.\n");
        return -4;
    }

    int num_classes = g_output->bytes;

    int best_class = 0;
    int best_raw = -128;

    for (int i = 0; i < num_classes; i++)
    {
        int raw = (int)g_output->data.int8[i];

        if (raw > best_raw)
        {
            best_raw = raw;
            best_class = i;
        }
    }

    int best_percent =
        ((best_raw - g_output->params.zero_point) * 100) / 256;

    if (best_percent < 0)
    {
        best_percent = 0;
    }
    else if (best_percent > 100)
    {
        best_percent = 100;
    }

    if (predicted_class != nullptr)
    {
        *predicted_class = best_class;
    }

    if (confidence_percent != nullptr)
    {
        *confidence_percent = best_percent;
    }

    /* Final demo mode:
     * Do not print all raw class scores here.
     * hal_entry.c prints the clean prediction and alert result.
     */

    return 0;
}

extern "C" int cpp_tflm_run_dummy(int *predicted_class,
                                  int *confidence_percent)
{
    if (!g_tflm_ready || g_input == nullptr)
    {
        rt_kprintf("TFLM ERROR: Model not initialized.\n");
        return -1;
    }

    static int8_t dummy_features[2379];

    for (unsigned int i = 0; i < g_input->bytes; i++)
    {
        dummy_features[i] = (int8_t)g_input->params.zero_point;
    }

    return cpp_tflm_run_features(dummy_features,
                                 g_input->bytes,
                                 predicted_class,
                                 confidence_percent);
}
