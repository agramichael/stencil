EXE=stencil.exe

CFLAGS= -Ofast -restrict -xAVX -std=c99 -Wall

$(EXE): %.exe : %.c
	mpiicc $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXE)
	\rm -f *.o
