ARCH = riscv64
ADDR_TYPE ?= uint64_t
PRIxBITS ?= PRIx64
PRIuBITS ?= PRIu64

AR=$(RV64_GNU_INSTALL)/bin/riscv64-unknown-linux-musl-ar
CC=$(RV64_GNU_INSTALL)/bin/riscv64-unknown-linux-musl-gcc

RDMADIR=$(abspath ../sst-elements/src/sst/elements/rdmaNic/)
APPDIR=$(RDMADIR)/tests/app/rdma/

CFLAGS=-I$(APPDIR)/include -I$(RDMADIR) -Wattributes -Wall -DADDR_TYPE=$(ADDR_TYPE)  -DPRIxBITS=$(PRIxBITS) -DPRIuBITS=$(PRIuBITS)
LIBS=-lrdma -Lriscv64

OBJS=riscv64/base.o riscv64/rdma.o


.PHONY: all clean

all: librdma.a
librdma.a: ${OBJS}
	$(AR) rcs riscv64/librdma.a $^

riscv64/rdma.o: $(APPDIR)/src/rdma.c $(APPDIR)/include/rdma.h $(RDMADIR)/rdmaNicHostInterface.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

riscv64/base.o: $(APPDIR)/src/base.c $(APPDIR)/include/base.h $(RDMADIR)/rdmaNicHostInterface.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf riscv64

