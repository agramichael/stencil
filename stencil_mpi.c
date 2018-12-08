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
  int ii,jj;              /* row and column indices for the grid */
  int k;                  /* stencil index */
  int kk;                 /* index for looping over ranks */
  int iter;               /* index for iterations */
  int num_iters;          /* number of times stencil is performed */
  int rank;               /* the rank of this process */
  int left;               /* the rank of the process to the left */
  int right;              /* the rank of the process to the right */
  int size;               /* number of processes in the communicator */
  int tag = 0;            /* default message tag */
  MPI_Status status;      /* struct used by MPI_Recv */
  int local_nx;           /* number of columns apportioned to this rank */
  int remote_nx;          /* number of columns apportioned to a remote rank */
  int nx, ny, niters;     /* command line arguments */
  float *tmp_image;       /* local grid at previous iteration */
  float *image;           /* local grid at current iteration     */
  float *sendbuf;         /* buffer to hold values to send */
  float *recvbuf;         /* buffer to hold received values */
  float *printbuf;        /* buffer to hold values for output */
  float *final_image;     /* final image */
  double tic, toc;        /* timer variables */
  int *sendcounts;
  int *displs;/* scatter and gather arrays */

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
  */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  /*
  ** determine local grid size
  */
  local_nx = calc_nx_from_rank(rank, size, nx);
  /*
  ** allocate space for:
  ** - the local grids (2 extra columns added for the halos)
  ** - buffers for message passing
  */
  tmp_image = (float*) _mm_malloc(sizeof(float) * (local_nx+2) * ny, 32);
  image = (float*) _mm_malloc(sizeof(float) * (local_nx+2) * ny, 32);
  sendbuf = (float*) _mm_malloc(sizeof(float) * ny, 32);
  recvbuf = (float*) _mm_malloc(sizeof(float) * ny, 32);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  remote_nx = calc_nx_from_rank(size-1, size, nx);
  printbuf = (float*) _mm_malloc(sizeof(float) * (remote_nx+2) * ny, 32);

  // allocate final image in master
  if (rank == MASTER) {
      final_image = (float*) _mm_malloc(sizeof(float) * nx * ny, 32);
      init_image(nx, ny, final_image);
  }

  //timing
  tic = wtime();

  // Scatter image
  sendcounts = (int *) _mm_malloc(sizeof(int) * size, 32);
  displs = (int *) _mm_malloc(sizeof(int) * size, 32);
  #pragma vector aligned
  for (kk = 0; kk < size; kk++) {
    remote_nx = calc_nx_from_rank(kk, size, nx);
    sendcounts[kk] = remote_nx*ny;
    displs[kk] = kk*local_nx*ny;
  }
  MPI_Scatterv(final_image, sendcounts, displs,
                 MPI_FLOAT, &image[ny], local_nx*ny,
                 MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  /*
  ** Perform 5-point stencil.
  */
  //num_iters = niters*2;
  #pragma vector aligned
  for (iter = 0; iter < niters; iter++) {

    // EXCHANGE HALOS from and to image
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
    ** compute new values of tmp_image using image
    */
    if (rank == MASTER) {
      // core
      for (ii = 1; ii < local_nx + 1; ii++) {
        for (jj = 0; jj < ny; jj++) {
          k = jj+ii*ny;
          tmp_image[k] = image[k] * (float) 0.6;
          if (ii > 1)    tmp_image[k] += image[k - ny] * (float) 0.1;
          tmp_image[k] += image[k + ny] * (float) 0.1;
          if (jj > 0)    tmp_image[k] += image[k - 1] * (float) 0.1;
          if (jj < ny-1) tmp_image[k] += image[k + 1] * (float) 0.1;
        }
      }
    }
    else if (rank == size - 1) {
      for (ii = 1; ii < local_nx + 1; ii++) {
        for (jj = 0; jj < ny; jj++) {
          k = jj+ii*ny;
          tmp_image[k] = image[k] * (float) 0.6;
          tmp_image[k] += image[k - ny] * (float) 0.1;
          if (ii < local_nx) tmp_image[k] += image[k + ny] * (float) 0.1;
          if (jj > 0)    tmp_image[k] += image[k - 1] * (float) 0.1;
          if (jj < ny-1) tmp_image[k] += image[k + 1] * (float) 0.1;
        }
      }
    }
    else {
      for (ii = 1; ii < local_nx + 1; ii++) {
        for (jj = 0; jj < ny; jj++) {
          k = jj+ii*ny;
          tmp_image[k] = image[k] * (float) 0.6;
          tmp_image[k] += image[k - ny] * (float) 0.1;
          tmp_image[k] += image[k + ny] * (float) 0.1;
          if (jj > 0)    tmp_image[k] += image[k - 1] * (float) 0.1;
          if (jj < ny-1) tmp_image[k] += image[k + 1] * (float) 0.1;
        }
      }
    }
    // EXCHANGE HALOS from and to tmp_image
    /* send to the left, receive from right */
    //put first column in buffer;
    for (jj = 0; jj < ny; jj++) {
      sendbuf[jj] = tmp_image[jj + ny];
    }
    MPI_Sendrecv(sendbuf, ny, MPI_FLOAT, left, tag,
		 recvbuf, ny, MPI_FLOAT, right, tag,
		 MPI_COMM_WORLD, &status);
    for (jj = 0; jj < ny; jj++) {
      tmp_image[jj + (local_nx + 1) * ny] = recvbuf[jj];
    }

    /* send to the right, receive from left */
    for (jj = 0; jj < ny; jj++) {
      sendbuf[jj] = tmp_image[jj + local_nx * ny];
    }
    MPI_Sendrecv(sendbuf, ny, MPI_FLOAT, right, tag,
		 recvbuf, ny, MPI_FLOAT, left, tag,
		 MPI_COMM_WORLD, &status);
    for (jj = 0; jj < ny; jj++) {
      tmp_image[jj] = recvbuf[jj];
    }
    /*
    ** compute new values of image using tmp_image
    */
    if (rank == MASTER) {
      // core
      for (ii = 1; ii < local_nx + 1; ii++) {
        for (jj = 0; jj < ny; jj++) {
          k = jj+ii*ny;
          image[k] = tmp_image[k] * (float) 0.6;
          if (ii > 1)    image[k] += tmp_image[k - ny] * (float) 0.1;
          image[k] += tmp_image[k + ny] * (float) 0.1;
          if (jj > 0)    image[k] += tmp_image[k - 1] * (float) 0.1;
          if (jj < ny-1) image[k] += tmp_image[k + 1] * (float) 0.1;
        }
      }
    }
    else if (rank == size - 1) {
      for (ii = 1; ii < local_nx + 1; ii++) {
        for (jj = 0; jj < ny; jj++) {
          k = jj+ii*ny;
          image[k] = tmp_image[k] * (float) 0.6;
          image[k] += tmp_image[k - ny] * (float) 0.1;
          if (ii < local_nx) image[k] += tmp_image[k + ny] * (float) 0.1;
          if (jj > 0)    image[k] += tmp_image[k - 1] * (float) 0.1;
          if (jj < ny-1) image[k] += tmp_image[k + 1] * (float) 0.1;
        }
      }
    }
    else {
      for (ii = 1; ii < local_nx + 1; ii++) {
        for (jj = 0; jj < ny; jj++) {
          k = jj+ii*ny;
          image[k] = tmp_image[k] * (float) 0.6;
          image[k] += tmp_image[k - ny] * (float) 0.1;
          image[k] += tmp_image[k + ny] * (float) 0.1;
          if (jj > 0)    image[k] += tmp_image[k - 1] * (float) 0.1;
          if (jj < ny-1) image[k] += tmp_image[k + 1] * (float) 0.1;
        }
      }
    }
  }

  // Gather image
  MPI_Gatherv(&image[ny], local_nx*ny, MPI_FLOAT,
                final_image, sendcounts, displs,
                MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  // timing
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

  MPI_Finalize();

  /* free up allocated memory */
  if (rank == MASTER) _mm_free(final_image);
  _mm_free(tmp_image);
  _mm_free(image);
  _mm_free(sendbuf);
  _mm_free(recvbuf);
  _mm_free(printbuf);

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
