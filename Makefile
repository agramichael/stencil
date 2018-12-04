EXE=stencil_mpi.exe

CFLAGS=-Wall -g -DDEBUG

all: $(EXE)

$(EXE): %.exe : %.c
	mpiicc $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXE)
	\rm -f *.o
