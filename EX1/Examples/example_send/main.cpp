// mpic++ -o mpi_vector main.cpp
// mpirun -np 2 ./mpi_vector

#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;
    std::vector<double> send_data(2*n, 0.0);
    std::vector<double> recv_data(2*n, 0.0);

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            send_data[i] = i+1;
        }

        // Send the vector to process 1
        MPI_Send(send_data.data(), n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process 0 sent data to process 1" << std::endl;
    }

    else if (rank == 1) {
        MPI_Status status;
        MPI_Recv(send_data.data() + 1, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        std::cout << "Process 1 received data from process 0" << std::endl;

        // Print recieved data
        std::cout << "Recieved data:" << std::endl;
        for (int i = 0; i < send_data.size(); ++i) {
            std::cout << send_data[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
