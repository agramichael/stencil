#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define MASTER 0
#define OUTPUT_FILE "stencil_mpi.pgm"

/* functions */
int calc_nrows_from_rank(int rank, int size, int NROWS);
void output_image(const char * file_name, const int NCOLS, const int NROWS, double **image);
double wtime(void);


int main(int argc, char* argv[])
{
  int ii,jj;             /* row and column indices for the grid */
  int i,j;               /* stencil indices */
  int kk;                /* index for looping over ranks */
  int start_row,end_row; /* rank dependent looping indices */
  int iter;              /* index for timestep iterations */
  int rank;              /* the rank of this process */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  int remote_nrows;      /* number of columns apportioned to a remote rank */
  double **tmp_image;            /* local temperature grid at time t - 1 */
  double **image;            /* local temperature grid at time t     */
  double *sendbuf;       /* buffer to hold values to send */
  double *recvbuf;       /* buffer to hold received values */
  double *printbuf;      /* buffer to hold values for printing */

  /* MPI_Init returns once it has started up processes */
  /* get size and rank */
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initialise problem dimensions from command line arguments
  int NCOLS = atoi(argv[1]);
  int NROWS = atoi(argv[2]);
  int ITERS = atoi(argv[3]);
  if (rank == MASTER) printf("NCOLS: %d NROWS: %d NITERS: %d \n", NCOLS, NROWS, ITERS);

  /*
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  /*
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_ncols = NCOLS;
  local_nrows = calc_nrows_from_rank(rank, size, NROWS);

  /*
  ** allocate space for:
  ** - the local grid (2 extra columns added for the halos)
  ** - we'll use local grids for current and previous timesteps
  ** - buffers for message passing
  */
  tmp_image = (double**) malloc(sizeof(double*) * (local_nrows + 2));
  for (ii = 0; ii < (local_nrows + 2); ii++) {
    tmp_image[ii] = (double*) malloc(sizeof(double) * local_ncols);
  }
  image = (double**) malloc(sizeof(double*) * (local_nrows + 2));
  for (ii = 0; ii < (local_nrows + 2); ii++) {
    image[ii] = (double*) malloc(sizeof(double) * local_ncols);
  }
  sendbuf = (double*) malloc(sizeof(double) * local_nrows);
  recvbuf = (double*) malloc(sizeof(double) * local_nrows);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  //remote_nrows = calc_nrows_from_rank(size-1, size, NROWS);
  printbuf = (double*) malloc(sizeof(double) * local_ncols);

  // master allocates space for final image.
  double **final_image = (double**) malloc(sizeof(double*) * NROWS);
  for (ii = 0; ii < NROWS; ii++) {
    final_image[ii] = (double*) malloc(sizeof(double) * NCOLS);
  }

  /*
  ** initialize the local grid for the present time (w):
  ** - set boundary conditions for any boundaries that occur in the local grid
  ** - initialize inner cells to the average of all boundary cells
  ** note the looping bounds for index jj is modified
  ** to accomodate the extra halo columns
  ** no need to initialise the halo cells at this point
  */
  // initialize all 0; no need to init halo cells.
  for (ii = 1; ii < local_nrows + 1; ii++) {
    for (jj = 0; jj < local_ncols; jj++) {
	    image[ii][jj] = 0.0;
      tmp_image[ii][jj] = 0.0;
    }
  }
  // initialize checkerboard
  int master_nrows = calc_nrows_from_rank(0, size, NROWS);
  for (j = 0; j < 8; j++) {
    for (i = 0; i < 8; i++) {
      if ((i + j) % 2 == 1) {
        for (jj = j * NCOLS / 8; jj < (j + 1) * NCOLS / 8; jj++) {
          for (ii = i * NROWS / 8; ii < (i + 1) * NROWS / 8; ii++) {
            if (local_nrows == master_nrows) {
              if ((ii > local_nrows * rank - 1) && (ii < local_nrows * (rank + 1))) {
                image[(ii % local_nrows) + 1][jj] = 100.0;
              }
            }
            else {
              if (ii > master_nrows * rank - 1) {
                int row = master_nrows * (rank + 1);
                if (ii < row) image[(ii % master_nrows) + 1][jj] = 100.0;
                else {
                  image[(ii % master_nrows) + 1 + master_nrows][jj] = 100.0;
                }
              }
            }
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double tic = wtime();

  /*
  ** Perform 5-point stencil.
  */
  // for(iter=0;iter<ITERS;iter++) {
  //   /*
  //   ** halo exchange for the local grids w:
  //   ** - first send to the left and receive from the right,
  //   ** - then send to the right and receive from the left.
  //   ** for each direction:
  //   ** - first, pack the send buffer using values from the grid
  //   ** - exchange using MPI_Sendrecv()
  //   ** - unpack values from the recieve buffer into the grid
  //   */
  //
  //   /* send to the left, receive from right */
  //   for(ii=0;ii<local_nrows;ii++)
  //     sendbuf[ii] = image[ii][1];
  //   MPI_Sendrecv(sendbuf, local_nrows, MPI_DOUBLE, left, tag,
	// 	 recvbuf, local_nrows, MPI_DOUBLE, right, tag,
	// 	 MPI_COMM_WORLD, &status);
  //   for(ii=0;ii<local_nrows;ii++)
  //     image[ii][local_ncols + 1] = recvbuf[ii];
  //
  //   /* send to the right, receive from left */
  //   for(ii=0;ii<local_nrows;ii++)
  //     sendbuf[ii] = image[ii][local_ncols];
  //   MPI_Sendrecv(sendbuf, local_nrows, MPI_DOUBLE, right, tag,
	// 	 recvbuf, local_nrows, MPI_DOUBLE, left, tag,
	// 	 MPI_COMM_WORLD, &status);
  //   for(ii=0;ii<local_nrows;ii++)
  //     image[ii][0] = recvbuf[ii];
  //
  //   /*
  //   ** copy the old solution into the u grid
  //   */
  //   for(ii=0;ii<local_nrows;ii++) {
  //     for(jj=0;jj<local_ncols + 2;jj++) {
	//        tmp_image[ii][jj] = image[ii][jj];
  //     }
  //   }
  //
  //   /*
  //   ** compute new values of w using u
  //   ** looping extents depend on rank, as we don't
  //   ** want to overwrite any boundary conditions
  //   */
  //   for(ii=1;ii<local_nrows-1;ii++) {
  //     if(rank == 0) {
	//        start_col = 2;
	//        end_col = local_ncols;
  //     }
  //     else if(rank == size -1) {
	//        start_col = 1;
	//        end_col = local_ncols - 1;
  //     }
  //     else {
	//        start_col = 1;
	//        end_col = local_ncols;
  //     }
  //     for(jj=start_col;jj<end_col + 1;jj++) {
	//        image[ii][jj] = (tmp_image[ii - 1][jj] + tmp_image[ii + 1][jj] + tmp_image[ii][jj - 1] + tmp_image[ii][jj + 1]) / 4.0;
  //     }
  //   }
  // }

  MPI_Barrier(MPI_COMM_WORLD);
  double toc = wtime();

  // Output timing
  if (rank == MASTER) {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");
  }

  /*
  ** at the end, write out the solution.
  ** for each row of the grid:
  ** - rank 0 first prints out its cell values
  ** - then it receives row values sent from the other
  **   ranks in order, and prints them.
  */

  for (ii = 1; ii < local_nrows + 1; ii++) {
    if(rank == MASTER) {
      //Build image.
      for (jj = 0; jj < local_ncols; jj++) {
         final_image[ii-1][jj] = image[ii][jj];
	       //printf("%6.2f ",image[ii][jj]);
      }
      for(kk=1;kk<size;kk++) { /* loop over other ranks */
	       //remote_nrows = calc_nrows_from_rank(kk, size, NROWS);
	       MPI_Recv(printbuf,local_ncols,MPI_DOUBLE,kk,tag,MPI_COMM_WORLD,&status);
	       for(jj=0; jj < local_ncols;jj++) {
           final_image[kk * local_nrows + (ii - 1)][jj] = printbuf[jj];
	         //printf("%6.2f ",printbuf[jj]);
	       }
      }
      //printf("\n");
    }
    else {
      MPI_Send(image[ii],local_ncols,MPI_DOUBLE,MASTER,tag,MPI_COMM_WORLD);
    }
  }

  // Save to stencil.pgm
  if(rank == MASTER) {
    output_image(OUTPUT_FILE, NCOLS, NROWS, final_image);
    //printf("\n");
  }
  /* don't forget to tidy up when we're done */
  MPI_Finalize();

  /* free up allocated memory */
  for(ii=0;ii<local_nrows;ii++) {
    free(tmp_image[ii]);
    free(image[ii]);
  }
  free(tmp_image);
  free(image);
  free(sendbuf);
  free(recvbuf);
  free(printbuf);

  /* and exit the program */
  return EXIT_SUCCESS;
}

void output_image(const char * file_name, const int NCOLS, const int NROWS, double **image) {

  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }
  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", NCOLS, NROWS);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for
  int ii, jj;
  double maximum = 0.0;
  for (jj = 0; jj < NCOLS; jj++) {
    for (ii = 0; ii < NROWS; ii++) {
      if (image[ii][jj] > maximum)
        maximum = image[ii][jj];
    }
  }

  // Output image, converting to numbers 0-255
  for (jj = 0; jj < NCOLS; jj++) {
    for (ii = 0; ii < NROWS; ii++) {
      fputc((char)(255.0*image[ii][jj]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}


int calc_nrows_from_rank(int rank, int size, int NROWS) {

  int nrows;

  nrows = NROWS / size;       /* integer division */
  if ((NROWS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += NROWS % size;  /* add remainder to last rank */
  }

  return nrows;
}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
