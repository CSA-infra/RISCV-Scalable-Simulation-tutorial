ARCH=x86
CFLAGS=-O3 -fopenmp
LDFLAGS=-lm

ifndef CC
$(error CC is not set)
endif

ifndef MPICC
$(error MPICC is not set)
endif

.PHONY: all clean
all : $(ARCH)/mha_OMP $(ARCH)/mha_MPI_OMP $(ARCH)/check_mpi $(ARCH)/hello_MPI_OMP $(ARCH)/gemm_OMP $(ARCH)/hello_MPI

$(ARCH)/mha_OMP : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_MPI_OMP : mha_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/check_mpi : check_mpi.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/hello_MPI_OMP : hello_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/gemm_OMP : gemm_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DTILE_SIZE=16 -o $@ $^ $(LDFLAGS)

$(ARCH)/hello_MPI : hello_MPI.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(ARCH)
