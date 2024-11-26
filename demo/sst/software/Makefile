ARCH_LIST= x86 riscv64

.PHONY: all clean
all clean:
	 @for arch in $(ARCH_LIST) ;					\
	 do													\
		  make -f $$arch.make $@ || exit 1;	\
	 done
