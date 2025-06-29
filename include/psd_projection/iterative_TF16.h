#ifndef PSD_PROJECTION_ITERATIVE_TF16_H
#define PSD_PROJECTION_ITERATIVE_TF16_H

#include <cublas_v2.h>

void projection_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int mat_size,
    const int mat_offset = 0,
    const int n_iter = 100
);

#endif // PSD_PROJECTION_ITERATIVE_TF16_H