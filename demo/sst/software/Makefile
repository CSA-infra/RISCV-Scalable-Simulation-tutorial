## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2025 IMEC. All rights reserved.
## ******************************************************************************

ARCH_LIST= x86 riscv64

.PHONY: all clean
all clean:
	 @for arch in $(ARCH_LIST) ;					\
	 do													\
		  make -f $$arch.make $@ || exit 1;	\
	 done
