stencil: stencil.c
	icc -std=c99 -Wall -xAVX $^ -o $@
