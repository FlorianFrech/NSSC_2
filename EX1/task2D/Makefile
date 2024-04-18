# Compiler
MPICXX = mpic++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math

# Executable name
EXEC = solverMPI

# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Rule to compile .cpp to .o
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Rule to link object files to executable
$(EXEC): $(OBJS)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

# Phony target to clean compiled files
clean:
	rm -f $(OBJS) $(EXEC)

# Run target
run: $(EXEC)
	mpirun -np 9 --oversubscribe ./$(EXEC) 2D benchmark 30 100 1.0 2.0