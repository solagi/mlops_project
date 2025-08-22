#include <advanced-machine-learning-2025_inferencing.h>
#include <Arduino_APDS9960.h>

// Constants and globals
#define FEATURE_MULTIPLIER 3 // Since we store R, G, B for each step
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * FEATURE_MULTIPLIER] = {0};

uint16_t r, g, b;

// Function to sample color sensor in real time
void sample_color_sensor() {
    for (uint32_t i = 0; i < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; i++) {
        APDS.readColor(r, g, b);
        features[i * FEATURE_MULTIPLIER + 0] = (float)r;
        features[i * FEATURE_MULTIPLIER + 1] = (float)g;
        features[i * FEATURE_MULTIPLIER + 2] = (float)b;
        delayMicroseconds(1000 * EI_CLASSIFIER_INTERVAL_MS);
    }
}

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");
    if (!APDS.begin()) {
        Serial.println("Error initializing APDS9960 sensor!");
        while (1);
    }
}

void loop() {
    ei_printf("Edge Impulse standalone inferencing (Arduino)\n");

    // Sample real-time color data
    sample_color_sensor();

    ei_impulse_result_t result = { 0 };
    signal_t features_signal;
    features_signal.total_length = sizeof(features) / sizeof(features[0]);
    features_signal.get_data = &raw_feature_get_data;

    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        delay(1000);
        return;
    }

    print_inference_result(result);

    delay(1000);
}

void print_inference_result(ei_impulse_result_t result) {
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#endif
}
