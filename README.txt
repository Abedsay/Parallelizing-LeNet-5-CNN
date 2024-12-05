How to compile and run each implementation.

First change the directory to the path you need (seq, mpi, omp, cuda) and use the Makefile function for compiling and then run.

Example for each implementation.

Sequential:     make clean
		make
		./main

MPI:    make clean
	make
	mpirun -np 8 ./main  //you can change 8 to any number of processes you need to try as long as you have 				    enough logical cores. Try "lscpu" in the terminal and look for CPU(s): (cores)

OpenMP:     make clean
	    make
	    export OMP_NUM_THREADS= 20 // same with mpi you can test with changing this number
	    ./main

CUDA:     make clean
	  make
	  ./main
