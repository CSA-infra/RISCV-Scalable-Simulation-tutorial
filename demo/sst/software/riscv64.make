ARCH= riscv64
CC=$(RV64_GNU_INSTALL)/bin/riscv64-unknown-linux-musl-gcc
MPICC=$(MVAPICH2_INSTALL_DIR)/bin/mpicc
CFLAGS=-O3 -fopenmp
LDFLAGS=-static -lm

.PHONY: all clean
all : $(ARCH)/mha_OMP_8 $(ARCH)/mha_OMP_16 $(ARCH)/mha_OMP_32 $(ARCH)/mha_OMP_64 \
   $(ARCH)/mha_MPI_OMP $(ARCH)/hello_MPI_OMP $(ARCH)/gemm_OMP $(ARCH)/hello_MPI

$(ARCH)/mha_OMP_8 : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DTILE_SIZE=8 -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_OMP_16 : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DTILE_SIZE=16 -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_OMP_32 : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DTILE_SIZE=32 -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_OMP_64 : mha_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DTILE_SIZE=64 -o $@ $^ $(LDFLAGS)

$(ARCH)/gemm_OMP : gemm_OMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DTILE_SIZE=16 -o $@ $^ $(LDFLAGS)

$(ARCH)/mha_MPI_OMP : mha_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/hello_MPI_OMP : hello_MPI_OMP.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/hello_MPI : hello_MPI.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	 rm -rf $(ARCH)
