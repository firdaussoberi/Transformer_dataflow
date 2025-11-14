#pragma once
#include "VecAdd.h"
#include "MatMul.h"
#include "Activations.h"
#include <cmath>
#include <algorithm>
#include <cstdint>   // int8_t

template<typename T, int rows, int hidden, int cols>
void bitlinear158b_forward(
    T input[rows][hidden],      // input matrix
    T weights[hidden][cols],    // real weights
    T biases[cols],             // bias vector
    T result[rows][cols]        // output matrix
) {
    // ===============================================================
    // 1. Compute beta (mean abs weight)
    // ===============================================================
    T beta = 0;
    for (int i = 0; i < hidden; i++)
        for (int j = 0; j < cols; j++)
            beta += std::abs(weights[i][j]);

    beta /= (hidden * cols);
    if (beta < (T)1e-8) beta = (T)1e-8;

    // ===============================================================
    // 2. Ternarize weights => store sign only (-1,0,+1)
    //    We use small ints so FPGA can optimize.
    // ===============================================================
    int8_t ternary_sign[hidden][cols];

    for (int i = 0; i < hidden; i++) {
        for (int j = 0; j < cols; j++) {
            T r = weights[i][j] / beta;
            if      (r >  0.5) ternary_sign[i][j] =  1;
            else if (r < -0.5) ternary_sign[i][j] = -1;
            else               ternary_sign[i][j] =  0;
        }
    }

    // ===============================================================
    // 3. Quantize activations to int8
    //    Store per-row scale = 1/gamma for dequant later.
    // ===============================================================
    int8_t x_int8[rows][hidden];
    T      scale_row[rows];

    for (int r = 0; r < rows; r++) {
        // find max abs
        T max_val = 0;
        for (int h = 0; h < hidden; h++) {
            T v = std::abs(input[r][h]);
            if (v > max_val) max_val = v;
        }
        if (max_val < (T)1e-8) max_val = (T)1e-8;

        T gamma = (T)127.0 / max_val;
        scale_row[r] = ((T)1.0) / gamma;  // dequant scale = 1/gamma

        for (int h = 0; h < hidden; h++) {
            T v = input[r][h] * gamma;
            v = std::max((T)-128.0, std::min((T)127.0, std::round(v)));
            x_int8[r][h] = (int8_t)v;
        }
    }

    // ===============================================================
    // 4. Integer MAC + scaling
    //    Compute tmp[r][c] = sum(x_int8 * ternary_sign)
    //    Then convert to float using scale_row * beta
    // ===============================================================
    T tmp[rows][cols];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {

            int32_t acc = 0;

            for (int h = 0; h < hidden; h++) {
                int8_t t = ternary_sign[h][c];
                if (t == 1)      acc += x_int8[r][h];
                else if (t == -1) acc -= x_int8[r][h];
            }

            // dequantize result:
            //   int_acc * (scale_row[r] * beta)
            tmp[r][c] = (T)acc * (scale_row[r] * beta);
        }
    }

    // ===============================================================
    // 5. Add bias (existing helper)
    // ===============================================================
    for (int r = 0; r < rows; r++) {
        vecadd<T, cols>(tmp[r], biases, result[r]);
    }
}
