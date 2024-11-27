ARCH=x86
CC=gcc-12
MPICC=mpicc
CFLAGS=-O3 -fopenmp
LDFLAGS=-lm

.PHONY: all clean
all : $(ARCH)/mha_OMP $(ARCH)/mha_MPI_OMP $(ARCH)/check_mpi

$(ARCH)/mha_OMP : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_MPI_OMP : mha_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/check_mpi : check_mpi.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(ARCH)
