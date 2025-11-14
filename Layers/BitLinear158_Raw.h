#pragma once
#include "VecAdd.h"
#include "MatMul.h"
#include "Activations.h"
#include <cmath>
#include <algorithm>

template<typename T, int rows, int hidden, int cols>
void bitlinear158b_forward(
    T input[rows][hidden],      // input matrix
    T weights[hidden][cols],    // real weights
    T biases[cols],             // bias vector
    T result[rows][cols]        // output matrix
) {
    // --- 1. Ternarize weights (like Python BitLinear158b) ---
    T ternary_weights[hidden][cols];
    T beta = 0;

    // compute mean absolute value of weights
    for(int i = 0; i < hidden; i++) {
        for(int j = 0; j < cols; j++) {
            beta += std::abs(weights[i][j]);
        }
    }
    beta /= (hidden * cols);
    if(beta < 1e-8) beta = 1e-8;

    // ternarize: -beta, 0, +beta
    for(int i = 0; i < hidden; i++) {
        for(int j = 0; j < cols; j++) {
            T val = weights[i][j] / beta;
            if(val > 0.5) ternary_weights[i][j] = beta;
            else if(val < -0.5) ternary_weights[i][j] = -beta;
            else ternary_weights[i][j] = 0;
        }
    }

    // --- 2. Quantize activations (hardcoded) ---
    T x_quant[rows][hidden];
    for(int i = 0; i < rows; i++) {
        // find max absolute value in the row
        T max_val = 0;
        for(int j = 0; j < hidden; j++)
            if(std::abs(input[i][j]) > max_val) max_val = std::abs(input[i][j]);
        if(max_val < 1e-8) max_val = 1e-8;

        T gamma = 127.0 / max_val;  // scale to [-127,127] like Python Q_b=8
        for(int j = 0; j < hidden; j++) {
            T val = input[i][j] * gamma;
            val = std::round(std::max(std::min(val, 127.0), -128.0));
            x_quant[i][j] = val / gamma;  // dequantized
        }
    }

    // --- 3. Compute linear output ---
    T tmp[rows][cols];
    matmul<T, rows, hidden, cols>(x_quant, ternary_weights, tmp);
    for(int i = 0; i < rows; i++) {
        vecadd<T, cols>(tmp[i], biases, result[i]);
    }
}
