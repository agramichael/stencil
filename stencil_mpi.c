#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define MASTER 0
#define OUTPUT_FILE "stencil_mpi.pgm"

/* functions */
void init_image(const int nx, const int ny, float *image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
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
  float *tmp_image;            /* local temperature grid at time t - 1 */
  float *image;            /* local temperature grid at time t     */
  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */
  float *printbuf;      /* buffer to hold values for printing */
  float *final_image;
  double tic, toc;

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
  tmp_image = (float*) malloc(sizeof(float) * (local_nx+2) * ny);
  image = (float*) malloc(sizeof(float) * (local_nx+2) * ny);
  sendbuf = (float*) malloc(sizeof(float) * ny);
  recvbuf = (float*) malloc(sizeof(float) * ny);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  remote_nx = calc_nx_from_rank(size-1, size, nx);
  printbuf = (float*) malloc(sizeof(float) * (remote_nx+2) * ny);

  // allocate final image in master
  if (rank == MASTER) {
      final_image = (float*) malloc(sizeof(float) * nx * ny);
      init_image(nx, ny, final_image);
  }

  //timing
  tic = wtime();

  int *sendcounts = (int *) malloc(sizeof(int) * size);
  int *displs = (int *) malloc(sizeof(int) * size);
  for (kk = 0; kk < size; kk++) {
    remote_nx = calc_nx_from_rank(kk, size, nx);
    sendcounts[kk] = remote_nx*ny;
    displs[kk] = kk*local_nx;
  }

  MPI_Scatterv(final_image, sendcounts, displs,
                 MPI_FLOAT, &image[ny], local_nx*ny,
                 MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  // if (rank == MASTER) {
  //   // master initialises own section
  //   for (ii = 1; ii < local_nx + 1; ii++) {
  //     for (jj = 0; jj < ny; jj++) {
  //       image[jj + ii * ny] = final_image[jj + (ii - 1) * ny];
  //     }
  //   }
  //   // master scatters other sections
  //   for (kk = 1; kk < size; kk++ ) {
  //     remote_nx = calc_nx_from_rank(kk, size, nx);
  //     for (ii = kk * local_nx; ii < remote_nx + kk * local_nx; ii++) {
  //       MPI_Send(&final_image[ii * ny],ny,MPI_FLOAT,kk,tag,MPI_COMM_WORLD);
  //     }
  //   }
  // }
  // else {
  //   // workers receive their section
  //   for (ii = 1; ii < local_nx + 1; ii++) {
  //     MPI_Recv(printbuf,ny,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD,&status);
  //     for (jj = 0; jj < ny; jj++) {
  //       image[jj + ii * ny] = printbuf[jj];
  //     }
  //   }
  // }

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
      sendbuf[jj] = image[jj + ny];
    }
    MPI_Sendrecv(sendbuf, ny, MPI_FLOAT, left, tag,
		 recvbuf, ny, MPI_FLOAT, right, tag,
		 MPI_COMM_WORLD, &status);
    for (jj = 0; jj < ny; jj++) {
      image[jj + (local_nx + 1) * ny] = recvbuf[jj];
    }

    /* send to the right, receive from left */
    for (jj = 0; jj < ny; jj++) {
      sendbuf[jj] = image[jj + local_nx * ny];
    }
    MPI_Sendrecv(sendbuf, ny, MPI_FLOAT, right, tag,
		 recvbuf, ny, MPI_FLOAT, left, tag,
		 MPI_COMM_WORLD, &status);
    for (jj = 0; jj < ny; jj++) {
      image[jj] = recvbuf[jj];
    }
    /*
    ** copy the old solution into the tmp_image grid
    */
    for (ii = 0; ii < local_nx + 2; ii++) {
      for (jj = 0; jj < ny; jj++) {
	       tmp_image[jj + ii * ny] = image[jj + ii * ny];
      }
    }

    /*
    ** compute new values of image using tmp_image
    */
    if (rank == MASTER) {
      //top left
      k = ny;
      image[k] = tmp_image[k] * (float) 0.6;
      image[k] += tmp_image[k+ny] * (float) 0.1;
      image[k] += tmp_image[k+1] * (float) 0.1;

      // bottom left
      k = 2 * ny - 1;
      image[k] = tmp_image[k] * (float) 0.6;
      image[k] += tmp_image[k+ny] * (float) 0.1;
      image[k] += tmp_image[k-1] * (float) 0.1;

      //left edge
      for (jj = 1; jj < ny - 1; jj++) {
        k = jj + ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k-1] * (float) 0.1;
        image[k] += tmp_image[k+1] * (float) 0.1;
      }

      // core
      for (ii = 2; ii < local_nx + 1; ii++) {
        // top edge of local grid (jj = 0)
        k = ii * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k+1] * (float) 0.1;
        // bottom edge of local grid (jj = ny - 1)
        k = (ny - 1) + ii * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k-1] * (float) 0.1;
        // core cells of local grid
        for (jj = 1; jj < ny - 1; jj++) {
          k = jj + ii * ny;
          image[k] = tmp_image[k] * (float) 0.6;
          image[k] += tmp_image[k-ny] * (float) 0.1;
          image[k] += tmp_image[k+ny] * (float) 0.1;
          image[k] += tmp_image[k-1] * (float) 0.1;
          image[k] += tmp_image[k+1] * (float) 0.1;
        }
      }
    }
    else if (rank == size - 1) {
      // top right
      k = local_nx * ny;
      image[k] = tmp_image[k] * (float) 0.6;
      image[k] += tmp_image[k-ny] * (float) 0.1;
      image[k] += tmp_image[k+1] * (float) 0.1;

      // bottom right
      k = (local_nx + 1) * ny - 1;
      image[k] = tmp_image[k] * (float) 0.6;
      image[k] += tmp_image[k-ny] * (float) 0.1;
      image[k] += tmp_image[k-1] * (float) 0.1;

      // right edge
      for (jj = 1; jj < ny - 1; jj++) {
        k = jj + local_nx * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k-1] * (float) 0.1;
        image[k] += tmp_image[k+1] * (float) 0.1;
      }
      // core
      for (ii = 1; ii < local_nx; ii++) {
        k = ii * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k+1] * (float) 0.1;
        // bottom edge of local grid (jj = ny - 1)
        k = (ny - 1) + ii * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k-1] * (float) 0.1;
        // core cells of local grid
        for (jj = 1; jj < ny - 1; jj++) {
          k = jj + ii * ny;
          image[k] = tmp_image[k] * (float) 0.6;
          image[k] += tmp_image[k-ny] * (float) 0.1;
          image[k] += tmp_image[k+ny] * (float) 0.1;
          image[k] += tmp_image[k-1] * (float) 0.1;
          image[k] += tmp_image[k+1] * (float) 0.1;
        }
      }
    }
    else {
      for (ii = 1; ii < local_nx + 1; ii++) {
        // top edge of local grid (jj = 0)
        k = ii * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k+1] * (float) 0.1;
        // bottom edge of local grid (jj = ny - 1)
        k = (ny - 1) + ii * ny;
        image[k] = tmp_image[k] * (float) 0.6;
        image[k] += tmp_image[k-ny] * (float) 0.1;
        image[k] += tmp_image[k+ny] * (float) 0.1;
        image[k] += tmp_image[k-1] * (float) 0.1;
        // core cells of local grid
        for (jj = 1; jj < ny - 1; jj++) {
          k = jj + ii * ny;
          image[k] = tmp_image[k] * (float) 0.6;
          image[k] += tmp_image[k-ny] * (float) 0.1;
          image[k] += tmp_image[k+ny] * (float) 0.1;
          image[k] += tmp_image[k-1] * (float) 0.1;
          image[k] += tmp_image[k+1] * (float) 0.1;
        }
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
  /* int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm) */
  MPI_Gatherv(&image[ny], local_nx*ny, MPI_FLOAT,
                final_image, sendcounts, displs,
                MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  if (rank == MASTER) {
    for (kk = 0; kk < size; kk++) {
      printf("%d, %d, %d\n", sendcounts[kk], displs[kk], kk);
    }
  }

  // GATHER
  // if (rank == MASTER) {
  //   for (ii = 1; ii < local_nx + 1; ii++) {
  //     for (jj = 0; jj < ny; jj++) {
  //        final_image[jj + (ii-1) * ny] = image[jj + ii * ny];
  //     }
  //   }
  //   for (kk = 1; kk < size; kk++) {
  //     remote_nx = calc_nx_from_rank(kk, size, nx);
  //     for (ii = kk * local_nx; ii < remote_nx + kk * local_nx; ii++) {
  //       MPI_Recv(printbuf,ny,MPI_FLOAT,kk,tag,MPI_COMM_WORLD,&status);
  //       for (jj = 0; jj < ny; jj++) {
  //         final_image[jj + ii * ny] = printbuf[jj];
  //       }
  //     }
  //   }
  // }
  // else {
  //   for (ii = 1; ii < local_nx + 1; ii++) {
  //     MPI_Send(&image[ii * ny],ny,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
  //   }
  // }

  toc = wtime();

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
  if (rank == MASTER) free(final_image);
  free(tmp_image);
  free(image);
  free(sendbuf);
  free(recvbuf);
  free(printbuf);

  /* and exit the program */
  return EXIT_SUCCESS;
}

void init_image(const int nx, const int ny, float *image) {
  int i,j,ii,jj;
  // Zero everything
  for (j = 0; j < ny; ++j) {
    for (i = 0; i < nx; ++i) {
      image[j+i*ny] = (float) 0.0;
    }
  }
  // Checkerboard
  for (j = 0; j < 8; ++j) {
    for (i = 0; i < 8; ++i) {
      for (jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2){
            image[jj+ii*ny] = (float) 100.0;
          }
        }
      }
    }
  }
}


void output_image(const char * file_name, const int nx, const int ny, float *image) {

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
  float maximum = 0.0;
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
