stencil: stencil.c
	icc -std=c99 -Wall $^ -o $@
gpr: stencil.c
		icc -std=c99 -pg -fopenmp -g -o stream.gprof stencil.c
		./stream.gprof 1024 1024 100
	  gprof -b -l stream.gprof gmon.out
