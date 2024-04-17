// mpic++ main.cpp -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math -o solverMPI
// mpirun -np 3 --oversubscribe ./solverMPI 2D test 6 100 1.0 2.0

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

void jacobi_iteration(int N, int rows, int cols, int num_procs, std::vector<double> &proc_grid_old, std::vector<double> &proc_grid_new, bool residual = false) {
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
void initialize_proc_grid(std::vector<double> &proc_grid, int rows, int cols, double fix_east, double fix_west, int cart_coords[2], int dims[2]) {
    // Skip first and last rows (= ghost layers)
    for (int i = 1; i < rows-1; ++i) {
        for (int j = 0; j < cols; ++j) {
            // West = 2nd column
            if (j == 1 && cart_coords[0] == 0) { // && cart_coords[0] == 0
                proc_grid[j + i * cols] = fix_west;
            }
            // East = 2nd last column
            if (j == cols - 2 && cart_coords[0] == dims[0] - 1) { // && cart_coords[0] == dims[0] - 1
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

/*Main function*/
int main(int argc, char *argv[]) try {
    // parse command line arguments
    auto opts = program_options::parse(argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get number of MPI Processors and rank
    int num_proc, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
      std::cout << "total number of MPI processes = " << num_proc << "" << std::endl;
      opts.print();
      std::cout << std::endl;
    }

    // Create cartesian topology
    int ndims = opts.mpi_mode;
    int dims[2]; // dims[0] = rows / y-direction, dims[1] = cols / x-direction
    std::array<int, 2> proc_dims = {0, 0};
    if (ndims == 1) {
      dims[0] = num_proc;
      dims[1] = 1;
    }
    else {
      // Check if num_proc is a prime number
      bool is_prime = true;
      for (int i = 2; i < num_proc; ++i) {
        if (num_proc % i == 0) {
            is_prime = false;
            break;
        }
        break;
      }
      if (is_prime) {
          dims[0] = num_proc;
          dims[1] = 1;
      }
      else {
        // Find the best partition
        MPI_Dims_create(num_proc, 2, std::data(proc_dims));
        dims[0] = proc_dims[0];
        dims[1] = proc_dims[1];
      }
    }
    if (my_rank == 0) {
      std::cout << "Processes along 1st dimension / y-direction: " << dims[0] << std::endl;
      std::cout << "Processes along 2nd dimension / x-direction: " << dims[1] << std::endl;
    }
    
    // Create cartesian 2D topology
    int periods = 0;
    int period_flags[ndims];
    period_flags[0] = 0;
    period_flags[0] = 0;
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, period_flags, periods, &comm_cart);

    // Get rank in cartesian topology
    int cart_rank, cart_coords[2];
    MPI_Comm_rank(comm_cart, &cart_rank);
    
    // Neighboring processes
    // Northern rank (my_rank + 1), southern rank (my_rank - 1) - direction 0
    int northern_rank, southern_rank;
    MPI_Cart_shift(comm_cart, 0, 1, &southern_rank, &northern_rank);
    // Western rank (my_rank - 1), eastern rank (my_rank + 1) - direction 1
    int western_rank, eastern_rank;
    MPI_Cart_shift(comm_cart, 1, 1, &western_rank, &eastern_rank);

    // Get coordinates in cartesian topology
    MPI_Cart_coords(comm_cart, cart_rank, ndims, cart_coords);
    std::cout << "Process " << cart_rank << "\tcoords: " << cart_coords[0] << ", " << cart_coords[1] << std::endl;

    // Calculate sizes of process grids
    int N = opts.N;
    int n_ghost_layers = 2;
    int rows_proc, cols_proc, pts_proc;

    // Check if remainders for rows and columns required
    bool rows_has_remainder = N % dims[0] != 0;
    bool cols_has_remainder = N % dims[1] != 0;

    // Adapt number of rows for procs in last row
    if (rows_has_remainder) {
      rows_proc = N / dims[0];
      if (cart_coords[0] == dims[0] - 1) {
        rows_proc = N - rows_proc * dims[0];
      }
    }
    else {
      rows_proc = N / dims[0] ;
    }
    // Check if remainder for cols required
    if (cols_has_remainder) {
      cols_proc = N / dims[1];
      if (cart_coords[1] == dims[1] - 1) {
        cols_proc = N - cols_proc * dims[1];
      }
    }
    else {
      cols_proc = N / dims[1] ;
    }
    // Print the number of remainder rows / cols
    if (cart_coords[0] == dims[0]-1 && cart_coords[1] == dims[1]-1) {
      std::cout << "Last process has " << rows_proc << " rows and " << cols_proc << " cols." << std::endl;
    }   

    // Add ghostlayers and calculate number of points per proc
    rows_proc += n_ghost_layers;
    cols_proc = cols_proc + n_ghost_layers;
    pts_proc = rows_proc * cols_proc;

    // Initialize the grid as a contiguous vector
    std::vector<double> proc_grid_1(pts_proc, 0.0);
    initialize_proc_grid(proc_grid_1, rows_proc, cols_proc, opts.fix_east, opts.fix_west, cart_coords, dims);
    std::vector<double> proc_grid_2 = proc_grid_1;

    if(cart_coords[0] == 1) {
      std::cout << "Coordinates: " << cart_coords[0] << ", " << cart_coords[1] << std::endl;
      print_proc_grid(proc_grid_1, rows_proc, cols_proc);
    }

    // Indices for boundary values in proc grid vector
    int north_boundary_start = cols_proc * (rows_proc - 1);
    int south_boundary_start = cols_proc;
    int west_boundary_start = 1;
    int east_boundary_start = cols_proc - 2;
    
    // Indices for recieving boundary values in ghost layer
    int south_ghost_start = 0; 
    int north_ghost_start = (rows_proc - 1) * cols_proc;

    // Number of entries to send and recieve - withouth west and east boundary values
    int n_rows = cols_proc; // number of elements to send/recieve to/from northern and southern processes
    int n_cols = rows_proc; // number of elements to send/recieve to/from western and eastern processes

    // Initialize vectors to store western and eastern boundary values if dims[1] > 0
    std::vector<double> send_western_boundary;
    std::vector<double> send_eastern_boundary;
    std::vector<double> receive_western_ghost;
    std::vector<double> receive_eastern_ghost;
    if(dims[1] > 0) {
      send_western_boundary.resize(n_cols, 0.0);
      send_eastern_boundary.resize(n_cols, 0.0);
      receive_western_ghost.resize(n_cols, 0.0);
      receive_eastern_ghost.resize(n_cols, 0.0);
    }
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform jacobi iterations
    for (int iter = 0; iter < opts.iters; ++iter) {
      // perform Jacobi iteration
      jacobi_iteration(opts.N, rows_proc, cols_proc, num_proc, proc_grid_1, proc_grid_2);

      // Print proc grid
      if (cart_coords[0] == 0 && cart_coords[1] == 0) {
        std::cout << "Iteration " << iter << ":" << std::endl;
        std::cout << "Process " << cart_coords[0] << ", " << cart_coords[1] << " has grid:" << std::endl;
        print_proc_grid(proc_grid_1, rows_proc, cols_proc);
      }
      MPI_Barrier(comm_cart);

      // Extract west and east boundary values if dims[1] > 0
      if (dims[1] > 1) {
        for(int i = 0; i<n_cols; ++i) {
          send_western_boundary[i] = proc_grid_2[i * cols_proc + west_boundary_start];
          send_eastern_boundary[i] = proc_grid_2[i * cols_proc + east_boundary_start];
        }
      }

      // Send and recieve boundary values from neighboring processes
      std::vector<MPI_Request> requests(4, MPI_REQUEST_NULL);
      if (dims[1] > 1) {
        requests.resize(8, MPI_REQUEST_NULL);
      }

      // Send and recieve rows
      // Send southern boundary - except for processes in first row
      if (cart_coords[0] > 0) {
        MPI_Isend(proc_grid_2.data() + south_boundary_start, n_rows, MPI_DOUBLE, southern_rank, 0, comm_cart, &requests[0]);
      }

      // Recieve in north ghost layer - except processes in last row
      if (cart_coords[0] < dims[0] - 1) {
        MPI_Irecv(proc_grid_2.data() + north_ghost_start, n_rows, MPI_DOUBLE, northern_rank, 0, comm_cart, &requests[1]);
      }

      // Send northern boundary - except for processes in last row
      if (cart_coords[0] < dims[0] - 1) {
        MPI_Isend(proc_grid_2.data() + north_boundary_start, n_rows, MPI_DOUBLE, northern_rank, 0, comm_cart, &requests[2]);
      }

      // Recieve in south ghost layer - except for processes in first row
      if (cart_coords[0] > 0) {
        MPI_Irecv(proc_grid_2.data() + south_ghost_start, n_rows, MPI_DOUBLE, southern_rank, 0, comm_cart, &requests[3]);
      }

      // Send and recieve columns
      if (dims[1] > 1) {
        // Send western boundary - except for processes in first column
        if (cart_coords[1] > 0) {
          MPI_Isend(send_western_boundary.data(), n_cols, MPI_DOUBLE, western_rank, 0, comm_cart, &requests[4]);
        }

        // Recieve in east ghost layer - except for processes in last column
        if (cart_coords[1] < dims[1] - 1) {
          MPI_Irecv(receive_eastern_ghost.data(), n_cols, MPI_DOUBLE, eastern_rank, 0, comm_cart, &requests[5]);
          // Insert recieved boundary values in ghost layer
          for (int i = 0; i < n_cols; ++i) {
            proc_grid_2[i * (east_boundary_start+1) + cols_proc] = receive_eastern_ghost[i]; // ghost layer is one element right = +1
          }
        }

        // Send eastern boundary - except for processes in last column
        if (cart_coords[1] < dims[1] - 1) {
          MPI_Isend(send_eastern_boundary.data(), n_cols, MPI_DOUBLE, eastern_rank, 0, comm_cart, &requests[6]);
        }

        // Recieve in west ghost layer - except for processes in first column
        if (cart_coords[1] > 0) {
          MPI_Irecv(receive_western_ghost.data(), n_cols, MPI_DOUBLE, western_rank, 0, comm_cart, &requests[7]);
          // Insert recieved boundary values in ghost layer
          for(int i = 0; i < n_cols; ++i) {
            proc_grid_2[i * (west_boundary_start-1) + cols_proc] = receive_western_ghost[i]; // ghost layer is one element left = -1
          }
        }

      MPI_Barrier(comm_cart);
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
      std::vector<double> merged_grid = merge_subgrids(num_proc, pts_proc, opts.N, proc_grid_2, cols_proc, rows_proc, comm_cart, rows_has_remainder);
      writeToFile(opts.name, merged_grid, opts.N);
    }

    // perform last Jacobi iteration with residual = true
    jacobi_iteration(opts.N, rows_proc, cols_proc, num_proc, proc_grid_1, proc_grid_2, true);
    
    // Send grid values to master process
    if (cart_rank != 0) {
      MPI_Send(proc_grid_2.data(), pts_proc, MPI_DOUBLE, 0, cart_rank, comm_cart);
    }

    // Process 0 receives final grid values and calculates norms
    if (cart_rank == 0) {
      std::vector<double> final_grid = merge_subgrids(num_proc, pts_proc, opts.N, proc_grid_2, cols_proc, rows_proc, comm_cart, rows_has_remainder);
      
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