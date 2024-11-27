ARCH ?= riscv64
ADDR_TYPE ?= uint64_t
PRIxBITS ?= PRIx64
PRIuBITS ?= PRIu64

AR=$(ARCH)-linux-gnu-ar
CC=$(ARCH)-linux-gnu-gcc-12

RDMADIR=$(abspath ../sst-elements/src/sst/elements/rdmaNic/)
APPDIR=$(RDMADIR)/tests/app/rdma/

CFLAGS=-I$(APPDIR)/include -I$(RDMADIR) -Wattributes -Wall -DADDR_TYPE=$(ADDR_TYPE)  -DPRIxBITS=$(PRIxBITS) -DPRIuBITS=$(PRIuBITS)
LIBS=-lrdma -L$(ARCH)

OBJS=$(ARCH)/base.o $(ARCH)/rdma.o


.PHONY: all clean

all: librdma.a
librdma.a: ${OBJS}
	$(AR) rcs $(ARCH)/librdma.a $^

$(ARCH)/rdma.o: $(APPDIR)/src/rdma.c $(APPDIR)/include/rdma.h $(RDMADIR)/rdmaNicHostInterface.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(ARCH)/base.o: $(APPDIR)/src/base.c $(APPDIR)/include/base.h $(RDMADIR)/rdmaNicHostInterface.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(ARCH)

