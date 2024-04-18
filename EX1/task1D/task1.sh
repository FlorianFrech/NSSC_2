#!/bin/bash

## job name

#SBATCH --job-name=benchmark1

## logfiles (stdout/stderr) %x=job-name %j=job-id

#SBATCH --output=stdout-%x.%j.log
#SBATCH --error=stderr-%x.%j.log

## resource requests 

#SBATCH --partition=nssc    # partition for 360.242 and 360.242
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks=40          # number of processes
#SBATCH --cpus-per-task=1   # number of cpus per process
#SBATCH --time=00:45:00     # set time limit to 1 minute

## load modules and compilation (still on the login node)

module load pmi/pmix-x86_64     # [P]rocess [M]anagement [I]nterface (required by MPI-Implementation)
module load mpi/openmpi-x86_64  # MPI implementation (including compiler-wrappers mpicc/mpic++)

mpic++ -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math main.cpp -o solverMPI

## submitting jobs (on the allocated resources)

# job: print the host name

mpirun hostname

# job: run the mpi-enabled executable passing command line arguments

mpi_mode="1D"
filename="benchmark"
resolutions="125 250 1000 2000"
iterations=1000

for resolution in $resolutions
    do
        for i in $(seq 1 40)
        do
        mpirun -np $i ./solverMPI $mpi_mode $filename $resolution $iterations -100 +100
    done
done

