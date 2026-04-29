#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <mpi.h>

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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    if (rank == 0) {
        cout << "MPI processes: " << num_procs << endl;

        ofstream results("results_mpi.csv", ios::app);
        if (num_procs == 1) {
            results << "N,Processes,Time\n";
        }
        results.close();
    }

    for (int N : sizes) {
        vector<vector<double>> A, B;

        if (rank == 0) {
            cout << "=== N=" << N << " ===" << endl;
            A = generateMatrix(N);
            B = generateMatrix(N);
        }

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int rows_per_proc = N / num_procs;
        int start_row = rank * rows_per_proc;
        int end_row = (rank == num_procs - 1) ? N : start_row + rows_per_proc;
        int local_rows = end_row - start_row;

        vector<double> A_local(local_rows * N);
        vector<double> B_local(N * N);

        if (rank == 0) {
            vector<double> A_flat(N * N);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    A_flat[i * N + j] = A[i][j];

            vector<double> B_flat(N * N);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    B_flat[i * N + j] = B[i][j];

            copy(B_flat.begin(), B_flat.end(), B_local.begin());

            vector<int> send_counts(num_procs), displs(num_procs);
            for (int i = 0; i < num_procs; i++) {
                int rows = (i == num_procs - 1) ? N - i * rows_per_proc : rows_per_proc;
                send_counts[i] = rows * N;
                displs[i] = i * rows_per_proc * N;
            }

            MPI_Scatterv(A_flat.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
                A_local.data(), local_rows * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }
        else {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                A_local.data(), local_rows * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }

        MPI_Bcast(B_local.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        auto start = chrono::high_resolution_clock::now();

        vector<double> C_local(local_rows * N, 0.0);

        for (int i = 0; i < local_rows; i++) {
            for (int k = 0; k < N; k++) {
                double aik = A_local[i * N + k];
                for (int j = 0; j < N; j++) {
                    C_local[i * N + j] += aik * B_local[k * N + j];
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

        vector<double> C_flat(N * N);
        vector<int> recv_counts(num_procs), displs(num_procs);

        for (int i = 0; i < num_procs; i++) {
            int rows = (i == num_procs - 1) ? N - i * rows_per_proc : rows_per_proc;
            recv_counts[i] = rows * N;
            displs[i] = i * rows_per_proc * N;
        }

        MPI_Gatherv(C_local.data(), local_rows * N, MPI_DOUBLE,
            C_flat.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Processes: " << num_procs << " Time: " << duration.count() << " sec" << endl;

            ofstream results("results_mpi.csv", ios::app);
            results << N << "," << num_procs << "," << duration.count() << "\n";
            results.close();
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    if (rank == 0) {
        cout << "\nSaved to results_mpi.csv" << endl;
    }

    return 0;
}