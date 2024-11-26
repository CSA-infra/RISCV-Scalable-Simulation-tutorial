ARCH=x86
CC=gcc-12
MPICC=mpicc
CFLAGS=-O3 -fopenmp
LDFLAGS=-lm

.PHONY: all clean
all : $(ARCH)/mha_openMP $(ARCH)/mpi_mha $(ARCH)/check_mpi

$(ARCH)/mha_openMP : mha_openMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/mpi_mha : mpi_mha.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ARCH)/check_mpi : check_mpi.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(ARCH)
