// mpic++ main2.cpp -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math -o solverMPI
// mpirun -np 3 --oversubscribe ./solverMPI 1D test 250 100 1.0 2.0

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <mpi.h>
#include <cmath>
#include <chrono>

namespace program_options {

struct Options {
  unsigned int mpi_mode;  
  std::string name;
  size_t N;
  size_t iters;
  double fix_west;
  double fix_east;  
  void print() const {
    std::printf("mpi_mode: %u\nD", mpi_mode);    
    std::printf("name: %s\n", name.c_str());
    std::printf("N: %zu\n", N);
    std::printf("iters: %zu\n", iters);
    std::printf("fix_west: %lf\n", fix_west);
    std::printf("fix_east: %lf\n", fix_east);    
  }
};

auto parse(int argc, char *argv[]) {
  if (argc != 7)
    throw std::runtime_error("unexpected number of arguments");
  Options opts;
  if (std::string(argv[1]) == std::string("1D"))
    opts.mpi_mode = 1;
  else if( std::string(argv[1]) == std::string("2D"))
    opts.mpi_mode = 2;
  else
   throw std::runtime_error("invalid parameter for mpi_mode (valid are '1D' and '2D')");
  opts.name = argv[2];
  if (std::sscanf(argv[3], "%zu", &opts.N) != 1 && opts.N >= 2)
    throw std::runtime_error("invalid parameter for N");
  if (std::sscanf(argv[4], "%zu", &opts.iters) != 1 && opts.iters != 0)
    throw std::runtime_error("invalid parameter for iters");
  if (std::sscanf(argv[5], "%lf", &opts.fix_west) != 1)
    throw std::runtime_error("invalid value for fix_west");
  if (std::sscanf(argv[6], "%lf", &opts.fix_east) != 1)
    throw std::runtime_error("invalid value for fix_east");  
  return opts;
}

} // namespace program_options

void jacobi_iteration(int N, int rows, int cols, int num_procs, int my_rank, std::vector<double> &proc_grid_old, std::vector<double> &proc_grid_new, bool residual = false) {
  auto h = 1.0 / (N - 1);
  auto h2 = h * h;
  // all interior points
  for (int i = 1; i < rows - 1; ++i) {
    for (int j = 2; j < cols - 2; ++j) {
      auto w = proc_grid_old[(i * cols) + (j - 1)];
      auto e = proc_grid_old[(i * cols) + (j + 1)];
      auto n = proc_grid_old[((i + 1) * cols) + j];
      auto s = proc_grid_old[((i - 1) * cols) + j];
      auto c = proc_grid_old[(i * cols) + j];
      if (!residual)
        proc_grid_new[(i * cols) + j] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
      else
        proc_grid_new[(i * cols) + j] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
    }
  }
  
  // isolating south boundary
  {
    int i = 1;
    for (int j = 2; j < cols - 2; ++j) {
      auto w = proc_grid_old[(i * cols) + (j - 1)];
      auto e = proc_grid_old[(i * cols) + (j + 1)];
      auto n = proc_grid_old[((i + 1) * cols) + j];
      auto s = n;
      auto c = proc_grid_old[(i * cols) + j];
      if (!residual)
        proc_grid_new[(i * cols) + j] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
      else
        proc_grid_new[(i * cols) + j] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
    }  
  }
    // isolating south boundary
  {
    int i = rows - 2;
    for (int j = 2; j < cols - 2; ++j) {
      auto w = proc_grid_old[(i * cols) + (j - 1)];
      auto e = proc_grid_old[(i * cols) + (j + 1)];
      auto s = proc_grid_old[((i - 1) * cols) + j];
      auto n = s;
      auto c = proc_grid_old[(i * cols) + j];
      if (!residual)
        proc_grid_new[(i * cols) + j] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
      else
        proc_grid_new[(i * cols) + j] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
    }
  }
} 

/*Set Dirichlet Boundary conditions on boundary grid points*/
void initialize_proc_grid(std::vector<double> &proc_grid, int rows, int cols, double fix_east, double fix_west) {
    // Skip first and last rows (= ghost layers)
    for (int i = 1; i < rows-1; ++i) {
        for (int j = 0; j < cols; ++j) {
            // West = 2nd column
            if (j == 1) {
                proc_grid[j + i * cols] = fix_west;
            }
            // East = 2nd last column
            if (j == cols - 2) {
                proc_grid[j + i * cols] = fix_east;
            }
        }
    }
}

/*Write to file*/
void writeToFile(std::string filename, std::vector<double> global_grid, int N) {
    std::ofstream csv;
    csv.open(filename + ".csv");
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        csv << global_grid[j + i * N] << " ";
      }
      csv << "\n";
    }
    csv.close();
}

/*Recieve subgrids and write to csv file*/
std::vector<double> merge_subgrids(int num_proc, int pts_proc, int global_N, std::vector<double> proc_grid_2,  int cols_proc, int rows_proc, MPI_Comm comm_cart, bool has_remainder) {
  // Initialize global grid vector
  std::vector<std::vector<double>> global_grid_recv(num_proc, std::vector<double>(pts_proc, 1.3));

  // if has_remainder resize the last subgrid array
  int pts_proc_remainder = 0;
  int rows_proc_remainder = 0;
  if (has_remainder) {
    rows_proc_remainder = global_N - (num_proc - 1) * (rows_proc-2) + 2;
    pts_proc_remainder = cols_proc * rows_proc_remainder;
    global_grid_recv[num_proc - 1].resize(pts_proc_remainder, 1.3);
  }

  // Write Master sub grid to global grid recieved
  for (int i = 0; i < pts_proc; ++i) {
    global_grid_recv[0][i] = proc_grid_2[i];
  }

  // Recieve subgrids from other processes
  for (int i = 1; i < num_proc; ++i) {
    MPI_Status status;
    if (has_remainder && i == num_proc - 1) {
      MPI_Recv(global_grid_recv[i].data(), pts_proc_remainder, MPI_DOUBLE, i, i, comm_cart, &status);
    }
    else {
      MPI_Recv(global_grid_recv[i].data(), pts_proc, MPI_DOUBLE, i, i, comm_cart, &status);
    }
  }

  // Merge subgrids - matrix to vector
  int global_grid_size = 0;
  if (has_remainder) {
    global_grid_size = (num_proc - 1) * pts_proc + pts_proc_remainder;
  }
  else {
    global_grid_size = num_proc * pts_proc;
  }
  std::vector<double> global_grid (global_grid_size, 3.0);
  for (int i = 0; i < num_proc; ++i) {
    if (has_remainder && i == num_proc - 1) {
      for (int j = 0; j < pts_proc_remainder; ++j) {
        global_grid[i * pts_proc + j] = global_grid_recv[i][j];
      }
    }
    else {
      for (int j = 0; j < pts_proc; ++j) {
      global_grid[i * pts_proc + j] = global_grid_recv[i][j];
      }
    }
  }
  
  // Remove ghost layers
  std::vector<double> global_grid_cleaned(global_N * global_N, 6.0);
  int p = 0;
  for (int k = 0; k < num_proc; ++k) {
    int start = k * pts_proc;
    if (has_remainder && k == num_proc - 1) {
      for (int i = 1; i < rows_proc_remainder-1; ++i) {
        for (int j = 1; j < cols_proc - 1; ++j) {
          global_grid_cleaned[p] = global_grid[start + j + i * cols_proc];
          p++;
        }
      }
    }
    else {
      for (int i = 1; i < rows_proc - 1; ++i) {
        for (int j = 1; j < cols_proc - 1; ++j) {
          global_grid_cleaned[p] = global_grid[start + j + i * cols_proc];
          p++;
        }
      }
    }
  }
  return global_grid_cleaned;
}

/*2 norm*/
double norm2(const std::vector<double> &x, int N) {
  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      sum += x[j + i * N] * x[j + i * N];
    }
  }
  return std::sqrt(sum);
}

/*Inf norm*/
double normInf(const std::vector<double> &x, int N) {
  double max = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (std::abs(x[j + i * N]) > max)
        max = std::abs(x[j + i * N]);
    }
  }
  return max;
}

/*Main function*/
int main(int argc, char *argv[]) try {
    // parse command line arguments
    auto opts = program_options::parse(argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int num_proc, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
      std::cout << "Num_Procs: " << num_proc << "" << std::endl;
      std::cout << "N: " << opts.N << std::endl;
      std::cout << "Iterations: " << opts.iters << std::endl;
      //opts.print();
    }

    // Create cartesian 1D topology
    int ndims = opts.mpi_mode;
    int dims[ndims];
    dims[0] = num_proc;
    int periods = 0;
    int period_flags[ndims];
    period_flags[0] = 0;

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, period_flags, periods, &comm_cart);

    int cart_rank, cart_coords;
    MPI_Comm_rank(comm_cart, &cart_rank);
    MPI_Cart_coords(comm_cart, cart_rank, ndims, &cart_coords);

    // Northern rank (my_rank + 1), southern rank (my_rank - 1)
    int northern_rank, southern_rank;
    MPI_Cart_shift(comm_cart, 0, 1, &southern_rank, &northern_rank);

    int N = opts.N;
    int n_ghost_layers = 2;
    int rows_proc, cols_proc, pts_proc;
    bool has_remainder = N % num_proc != 0;
    // Adapt number of rows for last process
    if (has_remainder) {
      rows_proc = N / num_proc;
      if (my_rank == num_proc - 1) {
        rows_proc = N - rows_proc * (num_proc-1);
      }
    }
    else {
      rows_proc = N / num_proc ;
    }
    if (my_rank == num_proc - 1 && has_remainder) {
      //std::cout << "Last process has " << rows_proc << " rows." << std::endl;
    }
    if (my_rank == 0) {
      //std::cout << "Processes have " << rows_proc << " rows." << std::endl;
    }
    rows_proc += n_ghost_layers;
    cols_proc = opts.N + n_ghost_layers;
    pts_proc = rows_proc * cols_proc;

    // Initialize the grid as a contiguous vector
    std::vector<double> proc_grid_1(pts_proc, 0.0);
    initialize_proc_grid(proc_grid_1, rows_proc, cols_proc, opts.fix_east, opts.fix_west);
    std::vector<double> proc_grid_2 = proc_grid_1;

    // indices for boundary values
    int north_boundary_start = cols_proc * (rows_proc - 1);
    int south_boundary_start = cols_proc;
    
    // Recieve boundary values in ghost layer
    int south_ghost_start = 0; 
    int north_ghost_start = (rows_proc - 1) * cols_proc;

    // Number of entries to send and recieve - withouth west and east boundary values
    int n = cols_proc;

    auto start = std::chrono::high_resolution_clock::now();

    // Perform jacobi iterations
    for (int iter = 0; iter <= opts.iters; ++iter) {
      // perform Jacobi iteration
      jacobi_iteration(opts.N, rows_proc, cols_proc, num_proc, my_rank, proc_grid_1, proc_grid_2);

      // Send and recieve boundary values from neighboring processes
      std::vector<MPI_Request> requests(4, MPI_REQUEST_NULL);

      // Send southern boundary - except for first process
      if (cart_rank > 0) {
        MPI_Isend(proc_grid_2.data() + south_boundary_start, n, MPI_DOUBLE, southern_rank, 0, comm_cart, &requests[0]);
      }

      // Recieve in north ghost layer - except for last process
      if (cart_rank < num_proc - 1) {
        MPI_Irecv(proc_grid_2.data() + north_ghost_start, n, MPI_DOUBLE, northern_rank, 0, comm_cart, &requests[1]);
      }

      // Send northern boundary - except for last process
      if (cart_rank < num_proc - 1) {
        MPI_Isend(proc_grid_2.data() + north_boundary_start, n, MPI_DOUBLE, northern_rank, 0, comm_cart, &requests[2]);
      }

      // Recieve in south ghost layer - except for first process
      if (cart_rank > 0) {
        MPI_Irecv(proc_grid_2.data() + south_ghost_start, n, MPI_DOUBLE, southern_rank, 0, comm_cart, &requests[3]);
      }
      
      // Wait for all processes to finish
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      // swap proc_grid_1 and proc_grid_2
      std::swap(proc_grid_1, proc_grid_2);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Send grid values to master process
    if (cart_rank != 0) {
      MPI_Send(proc_grid_2.data(), pts_proc, MPI_DOUBLE, 0, cart_rank, comm_cart);
    }

    // Process 0 receives grid values and writes to file
    if (cart_rank == 0) {
      std::vector<double> merged_grid = merge_subgrids(num_proc, pts_proc, opts.N, proc_grid_2, cols_proc, rows_proc, comm_cart, has_remainder);
      writeToFile(opts.name, merged_grid, opts.N);
    }

    // perform last Jacobi iteration with residual = true
    jacobi_iteration(opts.N, rows_proc, cols_proc, num_proc, my_rank, proc_grid_1, proc_grid_2, true);
    
    // Send grid values to master process
    if (cart_rank != 0) {
      MPI_Send(proc_grid_2.data(), pts_proc, MPI_DOUBLE, 0, cart_rank, comm_cart);
    }

    // Process 0 receives final grid values and calculates norms
    if (cart_rank == 0) {
      std::vector<double> final_grid = merge_subgrids(num_proc, pts_proc, opts.N, proc_grid_2, cols_proc, rows_proc, comm_cart, has_remainder);
      
      std::cout << "norm2 = " << norm2(final_grid, opts.N) << std::endl;
      std::cout << "normInf = " << normInf(final_grid, opts.N) << std::endl;
      std::cout << "time = " << std::chrono::duration<double>(end - start).count() * 1000 << " ms\n" <<std::endl;
    }

    MPI_Finalize();
    
    return EXIT_SUCCESS;
        
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}