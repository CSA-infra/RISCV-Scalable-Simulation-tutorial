#include <stdio.h>
#include <mpi.h>
#include <omp.h>


int main(int argc, char ** argv) {

   int n_ranks, rid;
   int n_threads = 0, tid = 0;


   printf("Initializing MPI\n");

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rid);

   printf("[rank %d] Entering OMP section\n", rid);

   #pragma omp parallel private(tid, n_threads)
   {
      n_threads = omp_get_num_threads();
      tid = omp_get_thread_num();
      printf("Hello from thread %d out of %d from process %d out of %d\n",
            tid, n_threads, rid, n_ranks);
   }

   MPI_Finalize();

   return 0;
}
