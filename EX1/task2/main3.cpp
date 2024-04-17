// mpic++ main3.cpp -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math -o solverMPI
// mpirun -np 3 --oversubscribe ./solverMPI 2D test 250 100 1.0 2.0

#include <array>
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

/*Jacobi Iteration*/
void jacobi_iteration(int N, int rows, int cols, int num_procs_x, int num_procs_y, std::array<int, 2> cart_coords,
                      std::vector<double> &proc_grid_old, std::vector<double> &proc_grid_new, bool residual = false) {
  auto h = 1.0 / (N - 1);
  auto h2 = h * h;
  /*
  Partition of the grid:
  
                BND Neumann North
  ------------------------------------------
  BND West | 2.1 |    2.3   | 2.2 | BND East
  
  BND West | 3.1 |    3.3   | 3.2 | BND East
  
  BND West | 1.1 |    1.3   | 1.2 | BND East
  ------------------------------------------ 
                BND Neumann South
  
  */
  
  // 1. Processes southern boundary
  if (cart_coords[0] == 0) {
    // 1.1. Processes on western boundary
    if (cart_coords[1] == 0) {
      int i = 1; // skip first ghost layer row
      // start in 3rd column (skip ghost layer and Dirichlet boundary on the left)
      for (int j = 2; j < cols - 1; ++j) {
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
      // Continue with 2nd row
      for (int i = 2; i < rows - 1; ++i) {
        for (int j = 2; j < cols - 1; ++j) {
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
    }
    // 1.2. Processes on eastern boundary
    if (cart_coords[1] == num_procs_x - 1) {
      int i = 1; // skip first ghost layer row
      // start in 2nd column (skip ghost layer and Dirichlet boundary on the right)
      for (int j = 1; j < cols - 2; ++j) {
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
      // Continue with 2nd row
      for (int i = 2; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 2; ++j) {
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
    }
    // 1.3. Interior southern boundary
    else {
      int i = 1; // skip first ghost layer row
      // skip ghost layer on the left and right
      for (int j = 1; j < cols - 1; ++j) {
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
      // Continue with 2nd row
      for (int i = 2; i < rows - 1; ++i) {
        // no boundary on the left or right thus cols - 1
        for (int j = 1; j < cols - 1; ++j) {
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
    }
  }

  // 2. Processes on northern boundary
  if (cart_coords[0] == num_procs_y - 1) {
    // 2.1. Processes on western boundary
    if (cart_coords[1] == 0) {
      // Skip southern ghost layer, iterate until 2nd last row (exluded)
      for (int i = 1; i < rows - 2; ++i) {
        // skip boundary and ghost layer on the left
        for (int j = 2; j < cols - 1; ++j) {
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
      // Continue with Neumann boundary row
      int i = rows - 2;
      // skip ghost layer and Dirichlet boundary on the left
      for (int j = 2; j < cols - 1; ++j) {
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
    // 2.2. Processes on eastern boundary
    if (cart_coords[1] == num_procs_x - 1) {
      // Skip southern ghost layer, iterate until 2nd last row (exluded)
      for (int i = 1; i < rows - 2; ++i) {
        // skip ghost layer on the right
        for (int j = 1; j < cols - 2; ++j) {
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
      // Continue with Neumann boundary row
      int i = rows - 2;
      // skip ghost layer and Dirichlet boundary on the left
      for (int j = 1; j < cols - 2; ++j) {
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
    // 2.3. Interior northern boundary
    else {
      // Skip southern ghost layer, iterate until 2nd last row (exluded)
      for (int i = 1; i < rows - 2; ++i) {
        // no boundary on the left or right thus cols - 1
        for (int j = 1; j < cols - 1; ++j) {
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
      // Continue with Neumann boundary row
      int i = rows - 2;
      // no boundary on the left or right thus cols - 1
      for (int j = 1; j < cols - 1; ++j) {
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

  // 3. Interior points on western and eastern boundaries
  else {
    // 3.1.Processes on western boundary
    if (cart_coords[1] == 0) {
      // Skip southern and northern ghost layer
      for (int i = 1; i < rows - 1; ++i) {
        // skip boundary and ghost layer on the left
        for (int j = 2; j < cols - 1; ++j) {
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
    }

    // 3.2. Processes on eastern boundary
    if (cart_coords[1] == num_procs_x - 1) {
      // Skip southern and northern ghost layer
      for (int i = 1; i < rows - 1; ++i) {
        // skip ghost and dirichlet layer on the right
        for (int j = 1; j < cols - 2; ++j) {
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
    }

    // 3.3. Interior points - no boundaries 
    else {
      // Skip southern and northern ghost layer
      for (int i = 1; i < rows - 1; ++i) {
        // skip left and right ghost layer
        for (int j = 1; j < cols - 1; ++j) {
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
    }
  }


  // all interior points
  for (int i = 1; i < rows - 1; ++i) {
    for (int j = 1; j < cols - 2; ++j) {
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

/*Initialize process grids with Dirichlet boundary conditions and zero*/
void initialize_proc_grid(std::vector<double> &proc_grid, int rows, int cols, double fix_east, double fix_west, std::array<int, 2> coords, int n_procs_x) {
    // Skip first and last rows (= ghost layers)
    for (int i = 1; i < rows-1; ++i) {
        for (int j = 0; j < cols; ++j) {
            // West = 2nd column
            if (j == 1 && coords[1] == 0) { // && cart_coords[0] == 0
                proc_grid[j + i * cols] = fix_west;
            }
            // East = 2nd last column
            if (j == cols - 2 && coords[1] == n_procs_x - 1) { // && cart_coords[0] == dims[0] - 1
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

/*Print process grid*/
void print_proc_grid(const std::vector<double> &x, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << x[j + i * cols] << " ";
    }
    std::cout << std::endl;
  }
}

/*Check if number of processes is a prime number*/
bool check_if_prime(int n) {
  for (int i = 2; i < n; ++i) {
    if (n % i == 0) {
      return false;
    }
  }
  return true;
}

/*Main function*/
int main(int argc, char *argv[]) try {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // parse command line arguments
    auto opts = program_options::parse(argc, argv);

    // Get number of MPI Processors and rank
    int num_proc, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Check for 1D or 2D mpi mode or prime numbers
    bool is_1D_mode = opts.mpi_mode == 1 || check_if_prime(num_proc);

    // Check for single process execution
    bool is_single_process_mode = num_proc == 1;

    // Let MIP choose number of processes per dimension
    std::array<int, 2> dims = {0, 0};
    MPI_Dims_create(num_proc, 2, std::data(dims));
    int num_proc_x = dims[0];
    int num_proc_y = dims[1];

    // Create a cartesian topology
    constexpr int n = 2;
    std::array<int, n> periods = {false, false};
    int reorder = true;
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periods), reorder, &comm_cart);

    // Get rank in cartesian topology
    int cart_rank;
    MPI_Comm_rank(comm_cart, &cart_rank);

    // Get ranks of neighbours in cartesian topology
    constexpr int displ = 1;
    std::array<int, n * 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    enum Direction : int { WEST = 0, EAST = 1, NORTH = 2, SOUTH = 3 };
    MPI_Cart_shift(comm_cart, 0, displ, &nb[SOUTH], &nb[NORTH]);
    MPI_Cart_shift(comm_cart, 1, displ, &nb[WEST], &nb[EAST]);

    // Get the coordinates in the cartesian topology
    std::array<int, n> coords = {-1, -1}; // coord [0] = y, coord [1] = x
    MPI_Cart_coords(comm_cart, cart_rank, n, std::data(coords));

    // Initialize grid variables
    int N = opts.N;
    int n_ghost_layers = 2;
    int rows_proc, cols_proc, pts_proc;

    // Check for remainders in rows and columns
    bool has_cols_remainder = N % num_proc_y != 0;
    bool has_rows_remainder = N % num_proc_x != 0;

    // Calculate the number of rows and columns in the grid for each process
    if (has_rows_remainder) {
      rows_proc = N / num_proc_y;
      if (coords[0] == num_proc_y - 1) {
        rows_proc = N - rows_proc * num_proc_y;
      }
    }
    else {
      rows_proc = N / num_proc_y;
    }
    // Check if remainder for cols required
    if (has_cols_remainder) {
      cols_proc = N / num_proc_x;
      if (coords[1] == num_proc_x - 1) {
        cols_proc = N - cols_proc * num_proc_x;
      }
    }
    else {
      cols_proc = N / dims[1] ;
    }
    if (is_1D_mode) {
      cols_proc = N;
    }
    if (is_single_process_mode) {
      rows_proc = N;
      cols_proc = N;
      // no ghost layers required
    }
    else {
      // Add ghost layers
      rows_proc += n_ghost_layers;
      cols_proc += n_ghost_layers;
    }
    pts_proc = rows_proc * cols_proc;

/*     if(coords[1] == 0 || coords[1] == num_proc_x - 1) {
      cols_proc -= 1; // remove one ghost layer column
    } */

    // Create MPI datatype for sending boundary values to ghost layers
    MPI_Datatype row;
    MPI_Type_vector(cols_proc, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Datatype col;
    MPI_Type_vector(rows_proc, 1, cols_proc, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Datatype row_1D;
    MPI_Type_vector(opts.N, 1, 1, MPI_DOUBLE, &row_1D);
    MPI_Type_commit(&row_1D);

    // Initialize the grid as a contiguous vector
    std::vector<double> proc_grid_1(pts_proc, 3.0);
    initialize_proc_grid(proc_grid_1, rows_proc, cols_proc, opts.fix_east, opts.fix_west, coords, num_proc_x);
    //std::vector<double> proc_grid_2 = proc_grid_1;
    std::vector<double> proc_grid_2(pts_proc, 0.0);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perfrom Jacobi Iteration
    for (int iter = 0; iter < opts.iters; ++iter) {
      // perform Jacobi iteration
      jacobi_iteration(N, rows_proc, cols_proc, num_proc_x, num_proc_y, coords, proc_grid_1, proc_grid_2);

      // Sending boundary values and recieving ghost values
      std::array<MPI_Request, n * 2 * 2> requests;
      
/*       if (my_rank == 0) std::cout << "Iteration " << iter << std::endl;
      MPI_Barrier(comm_cart);
      if (coords[0] == 0 && coords[1] == 1) print_proc_grid(proc_grid_2, rows_proc, cols_proc);
      MPI_Barrier(comm_cart);
      if(coords[0] == 1 && coords[1] == 1) print_proc_grid(proc_grid_2, rows_proc, cols_proc);
      MPI_Barrier(comm_cart); */

      if (coords[0] == 1 and coords[1] == 0) {
        std::cout << "Coords before sending, initialized grid: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_1, rows_proc, cols_proc);
      }
      MPI_Barrier(comm_cart);

      if (coords[0] == 1 and coords[1] == 1) {
        std::cout << "Coords before sending, initialized grid: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_1, rows_proc, cols_proc);
      }
      MPI_Barrier(comm_cart);

      // swap proc_grid_1 and proc_grid_2
      std::swap(proc_grid_1, proc_grid_2);

      // Send to western
      MPI_Isend(&proc_grid_1[1], 1, col, nb[WEST], WEST, comm_cart, &requests[0]);
      MPI_Irecv(&proc_grid_2[cols_proc-1], 1, col, nb[EAST], WEST, comm_cart, &requests[1]);

      // Send to eastern
      MPI_Isend(&proc_grid_1[cols_proc-2], 1, col, nb[EAST], EAST, comm_cart, &requests[2]);
      MPI_Irecv(&proc_grid_2[0], 1, col, nb[WEST], EAST, comm_cart, &requests[3]);

      // Send to southern
      MPI_Isend(&proc_grid_1[cols_proc], 1, row, nb[SOUTH], SOUTH, comm_cart, &requests[4]);
      MPI_Irecv(&proc_grid_2[cols_proc * (rows_proc - 1)], 1, row, nb[NORTH], SOUTH, comm_cart, &requests[5]);

      // Send to northern
      MPI_Isend(&proc_grid_1[cols_proc * (rows_proc - 2)], 1, row, nb[NORTH], NORTH, comm_cart, &requests[6]);
      MPI_Irecv(&proc_grid_2[0], 1, row, nb[SOUTH], NORTH, comm_cart, &requests[7]);

      // Wait for all processes to finish
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      if (coords[0] == 1 and coords[1] == 0) {
        std::cout << "Coords after recieving, proc grid 2: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_2, rows_proc, cols_proc);
      }
      MPI_Barrier(comm_cart);

      if (coords[0] == 1 and coords[1] == 1) {
        std::cout << "Coords after recieving, proc grid 2: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_2, rows_proc, cols_proc);
      }
      MPI_Barrier(comm_cart);
    }

/*     MPI_Barrier(comm_cart);
    if (coords[0] == 1 && coords[1] == 0) {
        std::cout << "Coords: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_2, rows_proc, cols_proc);
    }
    MPI_Barrier(comm_cart);
    
    if(coords[0] == 1 && coords[1] == 1) {
        std::cout << "Coords: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_2, rows_proc, cols_proc);
    }
    MPI_Barrier(comm_cart);

        if(coords[0] == 1 && coords[1] == 2) {
        std::cout << "Coords: " << coords[0] << " " << coords[1] << std::endl;
        print_proc_grid(proc_grid_2, rows_proc, cols_proc);
    }
    MPI_Barrier(comm_cart); */

/*     if (coords[0] == 0 && coords[1] == 0) {
      std::cout << "Final grid:" << std::endl;
      std::cout << "Coordinates: " << coords[0] << ", " << coords[1] << std::endl;
      print_proc_grid(proc_grid_1, rows_proc, cols_proc);
    }
    MPI_Barrier(comm_cart);
    if (coords[0] == 1 && coords[1] == 0) {
      std::cout << "Coordinates: " << coords[0] << ", " << coords[1] << std::endl;
      print_proc_grid(proc_grid_1, rows_proc, cols_proc);
    }
    MPI_Barrier(comm_cart);
    if (coords[0] == 2 && coords[1] == 0) {
      std::cout << "Coordinates: " << coords[0] << ", " << coords[1] << std::endl;
      print_proc_grid(proc_grid_1, rows_proc, cols_proc);
    }
    MPI_Barrier(comm_cart); */

    // End timer
    auto end = std::chrono::high_resolution_clock::now();

    MPI_Finalize();

    return EXIT_SUCCESS;
        
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}