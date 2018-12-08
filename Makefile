EXE=stencil.exe

CFLAGS= -Ofast -xAVX -std=c99 -Wall

$(EXE): %.exe : %.c
	mpiicc $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXE)
	\rm -f *.o
