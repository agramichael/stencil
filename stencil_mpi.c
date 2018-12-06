#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define MASTER 0
#define OUTPUT_FILE "stencil_mpi.pgm"

/* functions */
void init_image(const int nx, const int ny, double *image);
void output_image(const char * file_name, const int nx, const int ny, double *image);
int calc_nx_from_rank(int rank, int size, int nx);
double wtime(void);


int main(int argc, char* argv[])
{
  int ii,jj;             /* row and column indices for the grid */
  int i,j;               /* stencil indices */
  int k;
  int kk;                /* index for looping over ranks */
  int start_row,end_row; /* rank dependent looping indices */
  int iter;              /* index for timestep iterations */
  int rank;              /* the rank of this process */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nx;       /* number of rows apportioned to this rank */
  int remote_nx;      /* number of columns apportioned to a remote rank */
  int nx, ny, niters;
  double *tmp_image;            /* local temperature grid at time t - 1 */
  double *image;            /* local temperature grid at time t     */
  double *sendbuf;       /* buffer to hold values to send */
  double *recvbuf;       /* buffer to hold received values */
  double *printbuf;      /* buffer to hold values for printing */
  double *final_image;

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
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  niters = atoi(argv[3]);

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
  local_nx = calc_nx_from_rank(rank, size, nx);
  /*
  ** allocate space for:
  ** - the local grid (2 extra columns added for the halos)
  ** - we'll use local grids for current and previous timesteps
  ** - buffers for message passing
  */
  tmp_image = (double*) malloc(sizeof(double) * (local_nx+2) * ny);
  image = (double*) malloc(sizeof(double) * (local_nx+2) * ny);
  sendbuf = (double*) malloc(sizeof(double) * ny);
  recvbuf = (double*) malloc(sizeof(double) * ny);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  remote_nx = calc_nx_from_rank(size-1, size, nx);
  printbuf = (double*) malloc(sizeof(double) * (remote_nx+2) * ny);

  // allocates space for final image.
  final_image = (double*) malloc(sizeof(double) * nx * ny);

  /*
  ** initialize the local grid for the present time (w):
  ** - set boundary conditions for any boundaries that occur in the local grid
  ** - initialize inner cells to the average of all boundary cells
  ** note the looping bounds for index jj is modified
  ** to accomodate the extra halo columns
  ** no need to initialise the halo cells at this point
  */

  //Blank
  for (j = 0; j < ny; ++j) {
    for (i = 1; i < local_nx + 1; ++i) {
      image[j + i * ny] = 0.0;
      tmp_image[j + i * ny] = 0.0;
    }
  }
  // Checkerboard
  int master_nx = calc_nx_from_rank(0, size, nx);
  int start = master_nx * rank * ny;
  int end = master_nx * (rank + 1) * ny;

  for (j = 0; j < 8; ++j) {
    for (i = 0; i < 8; ++i) {
      if ((i+j)%2){
        for (jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
          for (ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
            int current = jj + ii * ny;
            if (local_nx == master_nx) {
              if ((current >= start) && (current < end)) {
                image[jj + ((ii%local_nx)+1) * ny] = 100.0;
              }
            }
            else {
              if (current >= start) {
                if (current < end) image[jj + ((ii%master_nx)+1) * ny] = 100.0;
                else {
                  image[jj + ((ii%master_nx) + 1 + master_nx) * ny] = 100.0;
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
  for (iter = 0; iter < niters*2; iter++) {
    /*
    ** halo exchange for the local grids (image):
    ** - first send to the left and receive from the right,
    ** - then send to the right and receive from the left.
    ** for each direction:
    ** - first, pack the send buffer using values from the grid
    ** - exchange using MPI_Sendrecv()
    ** - unpack values from the recieve buffer into the grid
    */

    /* send to the left, receive from right */
    //put first column in buffer;
    for (jj = 0; jj < ny; jj++) {
      sendbuf[jj] = image[jj];
    }
    MPI_Sendrecv(sendbuf, ny, MPI_DOUBLE, left, tag,
		 recvbuf, ny, MPI_DOUBLE, right, tag,
		 MPI_COMM_WORLD, &status);
    for (jj = 0; jj < ny; jj++) {
      image[jj + (local_nx + 1) * ny] = recvbuf[jj];
    }

    /* send to the right, receive from left */
    for (jj = 0; jj < ny; jj++) {
      sendbuf[jj] = image[jj + (local_nx + 1) * ny];
    }
    MPI_Sendrecv(sendbuf, ny, MPI_DOUBLE, right, tag,
		 recvbuf, ny, MPI_DOUBLE, left, tag,
		 MPI_COMM_WORLD, &status);
    for (jj = 0; jj < ny; jj++) {
      image[jj] = recvbuf[jj];
    }
    /*
    ** copy the old solution into the tmp_image grid
    */
    for (jj = 0; jj < ny; jj++) {
      for (ii = 0; ii < local_nx + 2; ii++) {
	       tmp_image[jj + ii * ny] = image[jj + ii * ny];
      }
    }

    /*
    ** compute new values of image using tmp_image
    */
    for (ii = 1; ii < local_nx + 1; ii++) {
      // top edge of local grid (jj = 0)
      k = ii * ny;
      image[k] = tmp_image[k] * 0.6;
      image[k] += tmp_image[k-ny] * 0.1;
      image[k] += tmp_image[k+ny] * 0.1;
      // image[k] += tmp_image[k-1] * 0.1;
      image[k] += tmp_image[k+1] * 0.1;
      // bottom edge of local grid (jj = ny - 1)
      k = (ny - 1) + ii * ny;
      image[k] = tmp_image[k] * 0.6;
      image[k] += tmp_image[k-ny] * 0.1;
      image[k] += tmp_image[k+ny] * 0.1;
      image[k] += tmp_image[k-1] * 0.1;
      // image[k] += tmp_image[k+1] * 0.1;

      // core cells of local grid
      for (jj = 1; jj < ny - 1; jj++) {
        k = jj + ii * ny;
        image[k] = tmp_image[k] * 0.6;
        image[k] += tmp_image[k-ny] * 0.1;
        image[k] += tmp_image[k+ny] * 0.1;
        image[k] += tmp_image[k-1] * 0.1;
        image[k] += tmp_image[k+1] * 0.1;
      }
    }
    // do left edge, top left and bottom left if MASTER (ii = 1)
    if (rank == MASTER) {

      //top left
      k = 0;
      image[k] = tmp_image[k] * 0.6;
      //image[k] += tmp_image[k-ny] * 0.1;
      image[k] += tmp_image[k+ny] * 0.1;
      //image[k] += tmp_image[k-1] * 0.1;
      image[k] += tmp_image[k+1] * 0.1;

      // bottom left
      k = 2 * ny - 1;
      image[k] = tmp_image[k] * 0.6;
      // image[k] += tmp_image[k-ny] * 0.1;
      image[k] += tmp_image[k+ny] * 0.1;
      image[k] += tmp_image[k-1] * 0.1;
      // image[k] += tmp_image[k+1] * 0.1;

      //left edge
      for (jj = 1; jj < ny - 1; jj++) {
        k = jj + ny;
        image[k] = tmp_image[k] * 0.6;
        // image[k] += tmp_image[k-ny] * 0.1;
        image[k] += tmp_image[k+ny] * 0.1;
        image[k] += tmp_image[k-1] * 0.1;
        image[k] += tmp_image[k+1] * 0.1;
      }
    }

    // do right edge, top right and bottom right if last (ii = local_nx)
    if (rank == size - 1) {
      // top right
      k = local_nx * ny;
      image[k] = tmp_image[k] * 0.6;
      image[k] += tmp_image[k-ny] * 0.1;
      // image[k] += tmp_image[k+ny] * 0.1;
      // image[k] += tmp_image[k-1] * 0.1;
      image[k] += tmp_image[k+1] * 0.1;

      // bottom right
      k = local_nx * ny + (ny - 1);
      image[k] = tmp_image[k] * 0.6;
      image[k] += tmp_image[k-ny] * 0.1;
      image[k] += tmp_image[k+ny] * 0.1;
      image[k] += tmp_image[k-1] * 0.1;
      image[k] += tmp_image[k+1] * 0.1;

      // right edge
      for (jj = 1; jj < ny - 1; jj++) {
        k = jj + local_nx * ny;
        image[k] = tmp_image[k] * 0.6;
        image[k] += tmp_image[k-ny] * 0.1;
        // image[k] += tmp_image[k+ny] * 0.1;
        image[k] += tmp_image[k-1] * 0.1;
        image[k] += tmp_image[k+1] * 0.1;
      }
    }
  }

  /*
  ** at the end, save the solution to final_image.
  ** for each column of the grid:
  ** - rank 0 first save its cell values
  ** - then it receives column values sent from the other
  **   ranks in order, and saves them.
  */

  for (ii = 1; ii < local_nx + 1; ii++) {
    if(rank == MASTER) {
      for (jj = 0; jj < ny; jj++) {
         final_image[jj + (ii-1) * ny] = image[jj + ii * ny];
      }
      for (kk = 1; kk < size; kk++) { /* loop over other ranks */
	       remote_nx = calc_nx_from_rank(kk, size, nx);
	       MPI_Recv(printbuf,ny,MPI_DOUBLE,kk,tag,MPI_COMM_WORLD,&status);
	       for (jj = 0; jj < ny; jj++) {
           final_image[jj + (kk * local_nx + (ii - 1)) * ny] = printbuf[jj];
	       }
      }
    }
    else {
      MPI_Send(&image[ii * ny],ny,MPI_DOUBLE,MASTER,tag,MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double toc = wtime();

  // Output timing
  if (rank == MASTER) {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");
  }

  //Save to stencil.pgm
  if(rank == MASTER) {
    output_image(OUTPUT_FILE, nx, ny, final_image);
  }
  /* don't forget to tidy up when we're done */
  MPI_Finalize();

  /* free up allocated memory */
  free(final_image);
  free(tmp_image);
  free(image);
  free(sendbuf);
  free(recvbuf);
  free(printbuf);

  /* and exit the program */
  return EXIT_SUCCESS;
}

void init_image(const int nx, const int ny, double *image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
    }
  }
  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2){
            image[jj+ii*ny] = 100.0;
          }
        }
      }
    }
  }
}


void output_image(const char * file_name, const int nx, const int ny, double *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  int i,j;
  double maximum = 0.0;
  for (j = 0; j < ny; ++j) {
    for (i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (j = 0; j < ny; ++j) {
    for (i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}


int calc_nx_from_rank(int rank, int size, int nx) {

  int ncols;

  ncols = nx / size;       /* integer division */
  if ((nx % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += nx % size;  /* add remainder to last rank */
  }

  return ncols;
}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
