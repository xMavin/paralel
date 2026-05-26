#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

using namespace std;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            cerr << "CUDA error: " << cudaGetErrorString(err__) << endl;              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

vector<double> generateMatrix(int n) {
    vector<double> matrix(n * n);
    mt19937 gen(42);
    uniform_int_distribution<> dist(0, 9);
    for (int i = 0; i < n * n; i++) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

__global__ void matMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

double runExperiment(int n, int blockDim, cudaStream_t stream) {
    vector<double> A = generateMatrix(n);
    vector<double> B = generateMatrix(n);
    vector<double> C(n * n, 0.0);

    size_t bytes = n * n * sizeof(double);

    double* dA, * dB, * dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), bytes, cudaMemcpyHostToDevice, stream));

    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((n + blockDim - 1) / blockDim, (n + blockDim - 1) / blockDim);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    matMulKernel << <gridSize, blockSize, 0, stream >> > (dA, dB, dC, n);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return ms / 1000.0;
}

void saveToCSV(const string& filename, const vector<int>& sizes, const vector<double>& times, int blockDim) {
    ofstream file(filename);
    file << "N,Time_sec,BlockSize\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        file << sizes[i] << "," << times[i] << "," << blockDim << "\n";
    }
    file.close();
}

void saveBlockResults(const string& filename, const vector<int>& blockSizes, const vector<double>& times, int n) {
    ofstream file(filename);
    file << "BlockSize,Time_sec,MatrixSize\n";
    for (size_t i = 0; i < blockSizes.size(); i++) {
        file << blockSizes[i] << "," << times[i] << "," << n << "\n";
    }
    file.close();
}

int main() {
    cout << "CUDA Matrix Multiplication Benchmark" << endl;

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "No CUDA GPU found!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cout << "\nExperiment 1: Varying matrix size (fixed block 16x16)" << endl;

    vector<int> sizes = { 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000 };
    vector<double> timeVsSize;

    for (int n : sizes) {
        cout << "  N = " << n << "... " << flush;
        double t = runExperiment(n, 16, stream);
        cout << t << " sec" << endl;
        timeVsSize.push_back(t);
    }

    saveToCSV("time_vs_size.csv", sizes, timeVsSize, 16);
    cout << "  Saved to time_vs_size.csv" << endl;

    cout << "\nExperiment 2: Varying block size (fixed matrix 1024x1024)" << endl;

    vector<int> blockSizes = { 4, 8, 12, 16, 20, 24, 28, 32 };
    vector<double> timeVsBlock;
    int fixedSize = 1024;

    for (int bs : blockSizes) {
        if (bs * bs > prop.maxThreadsPerBlock) {
            cout << "  Block " << bs << "x" << bs << "... SKIPPED" << endl;
            timeVsBlock.push_back(0);
            continue;
        }
        cout << "  Block " << bs << "x" << bs << "... " << flush;
        double t = runExperiment(fixedSize, bs, stream);
        cout << t << " sec" << endl;
        timeVsBlock.push_back(t);
    }

    saveBlockResults("time_vs_blocksize.csv", blockSizes, timeVsBlock, fixedSize);
    cout << "  Saved to time_vs_blocksize.csv" << endl;

    cout << "\nExperiment 3: Speedup (CPU vs GPU)" << endl;

    vector<double> cpuTimes;
    cout << "  Running CPU version..." << endl;

    for (int n : sizes) {
        cout << "    N = " << n << "... " << flush;

        auto start = chrono::high_resolution_clock::now();

        vector<double> A = generateMatrix(n);
        vector<double> B = generateMatrix(n);
        vector<double> C(n * n, 0.0);

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double aik = A[i * n + k];
                for (int j = 0; j < n; j++) {
                    C[i * n + j] += aik * B[k * n + j];
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        double t = chrono::duration<double>(end - start).count();
        cout << t << " sec" << endl;
        cpuTimes.push_back(t);
    }

    ofstream speedupFile("speedup_data.csv");
    speedupFile << "N,Time_CPU_sec,Time_GPU_sec,Speedup\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        double speedup = cpuTimes[i] / timeVsSize[i];
        speedupFile << sizes[i] << "," << cpuTimes[i] << "," << timeVsSize[i] << "," << speedup << "\n";
    }
    speedupFile.close();
    cout << "  Saved to speedup_data.csv" << endl;

    cout << "\nExperiments complete!" << endl;
    cout << "Generated files: time_vs_size.csv, time_vs_blocksize.csv, speedup_data.csv" << endl;

    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}