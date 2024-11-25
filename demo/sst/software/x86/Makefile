CC=gcc-12
MPICC=mpicc
CFLAGS=-g3
LDFALGS=-fopenmp -lm

.PHONY: all clean
all : mha mpi_mha test_tp

mha : mha.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFALGS)

mpi_mha : mpi_mha.c
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFALGS)

test_tp : test_tp.c
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFALGS)

clean:
	rm -f $(TARGET)
