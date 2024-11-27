ARCH= riscv64
CC=$(RV64_GNU_INSTALL)/bin/riscv64-unknown-linux-musl-gcc
MPICC=/home/lenorm62/vlsid-tutorial/external/mvapich2-2.3.7-1/install/bin/mpicc
CFLAGS=-O3 -fopenmp
LDFLAGS=-static -lm

.PHONY: all clean
all : $(ARCH)/mha_OMP $(ARCH)/mha_MPI_OMP

$(ARCH)/mha_OMP : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_MPI_OMP : mha_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	 rm -rf $(ARCH)
