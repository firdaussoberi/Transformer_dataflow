#pragma once
#include <cmath>

template<typename T, int channels, int size>
void rms_norm(
    T input[channels][size], 
    T epsilon, 
    T gamma[size], 
    T result[channels][size]
) {
rms_norm_outer_loop:
    for (int i = 0; i < channels; i++) {
        T sum_squares = 0.0;
rms_norm_sum_loop:
        for (int j = 0; j < size; j++) {
            T val = (T) input[i][j];
            sum_squares += val * val;
        }
        T rms = hls::sqrt(sum_squares / size + epsilon); // RMS computation
rms_norm_result_loop:
        for (int j = 0; j < size; j++) {
            result[i][j] = (input[i][j] * gamma[j]) / rms;
        }
    }
}
