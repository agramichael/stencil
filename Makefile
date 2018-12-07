EXE=stencil_mpi.exe

CFLAGS=-std=c99 -Wall -g -DDEBUG

all: $(EXE)

$(EXE): %.exe : %.c
	mpiicc $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXE)
	\rm -f *.o

stencil: stencil.c
	icc -std=c99 -Wall -xAVX $^ -o $@
