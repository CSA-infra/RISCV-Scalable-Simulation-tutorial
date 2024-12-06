ARCH= riscv64
CC=$(RV64_GNU_INSTALL)/bin/riscv64-unknown-linux-musl-gcc
MPICC=$(MVAPICH2_INSTALL_DIR)/bin/mpicc
CFLAGS=-O3 -fopenmp
LDFLAGS=-static -lm

.PHONY: all clean
all : $(ARCH)/mha_OMP $(ARCH)/mha_MPI_OMP $(ARCH)/hello_MPI_OMP

$(ARCH)/mha_OMP : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_MPI_OMP : mha_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/hello_MPI_OMP : hello_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	 rm -rf $(ARCH)
