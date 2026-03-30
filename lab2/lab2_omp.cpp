#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

vector<vector<double>> generateMatrix(int N) {
    vector<vector<double>> matrix(N, vector<double>(N));
    mt19937 gen(42);
    uniform_int_distribution<> dist(0, 9);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = dist(gen);
    return matrix;
}

vector<vector<double>> multiplyMatrices(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int N)
{
    vector<vector<double>> C(N, vector<double>(N, 0.0));

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double aik = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }

    return C;
}

int main() {
    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> threads = { 1, 2, 4, 8 };
    cout << "OpenMP: max threads = " << omp_get_max_threads() << endl;

    ofstream results("results.csv");
    results << "N,Threads,Time\n";

    for (int N : sizes) {
        cout << "=== SIZE " << N << " ===" << endl;

        auto A = generateMatrix(N);
        auto B = generateMatrix(N);

        for (int t : threads) {
            omp_set_num_threads(t);

            auto start = chrono::high_resolution_clock::now();
            auto C = multiplyMatrices(A, B, N);
            auto end = chrono::high_resolution_clock::now();

            chrono::duration<double> duration = end - start;
            cout << "Threads: " << t << " Time: " << duration.count() << " sec" << endl;
            results << N << "," << t << "," << duration.count() << "\n";
        }
        cout << endl;
    }

    results.close();
    cout << "DONE! Results saved to results.csv" << endl;

    return 0;
}