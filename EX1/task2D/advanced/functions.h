# ifndef functions_h
# define functions_h

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <mpi.h>
#include <cmath>
#include <chrono>
#include <iomanip>

// Jacobi solver - single process

// Jacobi solver - 1D
void jacobi_iteration_1D(int N, int rows, int cols, int num_procs,
                        std::vector<double> &proc_grid_old, std::vector<double> &proc_grid_new, bool residual = false) 

// Jacobi solver - 2D

// Initialize grid - single process

// Initialize subgrids

// Merge subgrids - 1D

// Merge Subgrids - 2D

// 2 norm

// Inf norm

// Print process grid

// Check if number of processes is a prime number

double norm2(const std::vector<double> &x, int N);

#endif