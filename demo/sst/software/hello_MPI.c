#include <stdio.h>
#include <mpi.h>

int main(int argc, char ** argv) {

   int n_ranks, rid;

   printf("Initializing MPI\n");

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rid);

   printf("Hello from process %d out of %d\n", rid, n_ranks);

   MPI_Finalize();

   return 0;
}
