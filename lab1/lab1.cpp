#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) 
{
    

    if (argc != 4) {
        cerr << "Incorrect number of arguments\n";
        return 1;
    }

    ifstream fileA(argv[1]);
    if (!fileA.is_open()) {
        cerr << "Error: Failed to open file " << argv[1] << "\n";
        return 1;
    }

    int n;
    fileA >> n; 

    vector<vector<double>> A(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            fileA >> A[i][j];
    fileA.close();

    ifstream fileB(argv[2]);
    if (!fileB.is_open()) {
        cerr << "Error: Failed to open file " << argv[2] << "\n";
        return 1;
    }

    int m;
    fileB >> m; 
    if (m != n) {
        cerr << "Error: matrix sizes do not match!\n";
        return 1;
    }

    vector<vector<double>> B(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            fileB >> B[i][j];
    fileB.close();

    vector<vector<double>> C(n, vector<double>(n, 0.0));

    auto start = high_resolution_clock::now();

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double aik = A[i][k];
            for (int j = 0; j < n; j++)
                C[i][j] += aik * B[k][j];
        }

    auto end = high_resolution_clock::now();
    double time = duration<double>(end - start).count();

    ofstream fileRes(argv[3]);
    if (!fileRes.is_open()) {
        cerr << "Error: Failed to create file " << argv[3] << "\n";
        return 1;
    }

    fileRes << fixed << setprecision(6);
    for (const auto& row : C) {
        for (double val : row)
            fileRes << val << " ";
        fileRes << "\n";
    }
    fileRes.close();

    cout << "Matrix size: " << n << "x" << n << "\n";
    cout << "Elements count: " << n * n << "\n";
    cout << "Time: " << time << " sec\n";

    return 0;
}