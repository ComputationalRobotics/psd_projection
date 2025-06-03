#ifndef PSD_PROJECTION_UTILS_H
#define PSD_PROJECTION_UTILS_H

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

// Print an n×n matrix of doubles
void printMatrixDouble(const double* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<double> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of floats
void printMatrixFloat(const float* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<float> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of __half
void printMatrixHalf(const __half* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<__half> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(__half), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            float v = __half2float(hM[i*n + j]);
            std::cout << std::fixed << std::setprecision(6)
                      << v << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

#endif // PSD_PROJECTION_UTILS_H