ARCH= riscv64
CC=$(RV64_GNU_INSTALL)/bin/riscv64-unknown-linux-musl-gcc
CFLAGS=-O3 -fopenmp
LDFLAGS=-static -lm

.PHONY: all clean
all : $(ARCH)/mha_openMP

$(ARCH)/mha_openMP : mha_openMP.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	 rm -rf $(ARCH)
