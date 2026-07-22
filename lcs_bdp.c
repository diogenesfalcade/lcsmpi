#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// #define DEBUGMATRIX

typedef unsigned short mtype;

// tile size
#define TILE_SIZE 128

/* Read sequence from a file to a char vector.
 Filename is passed as parameter */

char* read_seq(char *fname) {
	//file pointer
	FILE *fseq = NULL;
	//sequence size
	long size = 0;
	//sequence pointer
	char *seq = NULL;
	//sequence index
	int i = 0;

	//open file
	fseq = fopen(fname, "rt");
	if (fseq == NULL ) {
		printf("Error reading file %s\n", fname);
		exit(1);
	}

	//find out sequence size to allocate memory afterwards
	fseek(fseq, 0L, SEEK_END);
	size = ftell(fseq);
	rewind(fseq);

	//allocate memory (sequence)
	seq = (char *) calloc(size + 1, sizeof(char));
	if (seq == NULL ) {
		printf("Erro allocating memory for sequence %s.\n", fname);
		exit(1);
	}

	//read sequence from file
	while (!feof(fseq)) {
		seq[i] = fgetc(fseq);
		if ((seq[i] != '\n') && (seq[i] != EOF))
			i++;
	}
	//insert string terminator
	seq[i] = '\0';

	//close file
	fclose(fseq);

	//return sequence pointer
	return seq;
}

mtype ** allocateScoreMatrix(int sizeA, int sizeB) {
    int i;
    //Allocate memory for LCS score matrix
    mtype **scoreMatrix = (mtype **) malloc((sizeB + 1) * sizeof(mtype *));
    for (i = 0; i <= sizeB; i++)
        scoreMatrix[i] = (mtype *) calloc((sizeA + 1), sizeof(mtype));
    return scoreMatrix;
}

void initScoreMatrix(mtype **scoreMatrix, int sizeA, int sizeB) {
    int i;
    //Fill first line of LCS score matrix with zeroes
    for (i = 0; i <= sizeB; i++)
        scoreMatrix[i][0] = 0;
    //Do the same for the first collumn
    for (i = 0; i <= sizeA; i++)
        scoreMatrix[0][i] = 0;
}

int LCS(mtype **scoreMatrix, int sizeA, int sizeB, char *seqA, char *seqB) {
    // calculate how many blocks (tiles) are needed in both dimensions
    int nBlocksX = (sizeA + TILE_SIZE - 1) / TILE_SIZE;
    int nBlocksY = (sizeB + TILE_SIZE - 1) / TILE_SIZE;

    // start the OpenMP parallel region
    #pragma omp parallel
    {
        // only one thread creates the tasks
        #pragma omp single
        {
            for (int by = 0; by < nBlocksY; by++) {
                for (int bx = 0; bx < nBlocksX; bx++) {

                    // each block becomes an OpenMP task, processed when dependencies are satisfied
                    // dependencies ensure we respect LCS update order:
                    // - depends on the block above - same column
                    // - depends on the block to the left - same row
                    #pragma omp task firstprivate(bx, by) \
                            depend(in: scoreMatrix[by*TILE_SIZE][(bx-1)*TILE_SIZE], \
                            scoreMatrix[(by-1)*TILE_SIZE][bx*TILE_SIZE]) \
                            depend(out: scoreMatrix[by*TILE_SIZE][bx*TILE_SIZE])
                    {
                        // define row and column bounds for this block
                        int row_start = by * TILE_SIZE + 1;
                        int row_end   = (by + 1) * TILE_SIZE;
                        if (row_end > sizeB) row_end = sizeB;

                        int col_start = bx * TILE_SIZE + 1;
                        int col_end   = (bx + 1) * TILE_SIZE;
                        if (col_end > sizeA) col_end = sizeA;

                        // standard LCS update inside the block
                        for (int i = row_start; i <= row_end; i++) {
                            for (int j = col_start; j <= col_end; j++) {
                                if (seqA[j - 1] == seqB[i - 1]) {
                                    /* if elements in both sequences match,
				                    the corresponding score will be the score from
				                    previous elements + 1*/
                                    scoreMatrix[i][j] = scoreMatrix[i - 1][j - 1] + 1;
                                } else {
                                    /* else, pick the maximum value (score) from left and upper elements*/
                                    scoreMatrix[i][j] = max(scoreMatrix[i - 1][j], scoreMatrix[i][j - 1]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return scoreMatrix[sizeB][sizeA];
}

void printMatrix(char * seqA, char * seqB, mtype ** scoreMatrix, int sizeA, int sizeB) {
    int i, j;

    printf("Score Matrix:\n");
    printf("========================================\n");

    printf("    ");
    printf("%5c   ", ' ');
    for (j = 0; j < sizeA; j++)
        printf("%5c   ", seqA[j]);
    printf("\n");

    for (i = 0; i < sizeB + 1; i++) {
        if (i == 0)
            printf("    ");
        else
            printf("%c   ", seqB[i - 1]);
        for (j = 0; j < sizeA + 1; j++) {
            printf("%5d   ", scoreMatrix[i][j]);
        }
        printf("\n");
    }
    printf("========================================\n");
}

void freeScoreMatrix(mtype **scoreMatrix, int sizeB) {
    for (int i = 0; i <= sizeB; i++)
        free(scoreMatrix[i]);
    free(scoreMatrix);
}

int main(int argc, char ** argv) {
    
    // measure time
    struct timespec t_start, t_read, t_alloc, t_lcs_start, t_lcs_end, t_free, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // sequence pointers for both sequences
    char *seqA, *seqB;

    // sizes of both sequences
    int sizeA, sizeB;

    clock_gettime(CLOCK_MONOTONIC, &t_read);
    //read both sequences
    seqA = read_seq("fileA.in");
    seqB = read_seq("fileB.in");

    //find out sizes
    sizeA = strlen(seqA);
    sizeB = strlen(seqB);

    // allocate LCS score matrix
    clock_gettime(CLOCK_MONOTONIC, &t_alloc);
    mtype **scoreMatrix = allocateScoreMatrix(sizeA, sizeB);

    //initialize LCS score matrix
    initScoreMatrix(scoreMatrix, sizeA, sizeB);

    clock_gettime(CLOCK_MONOTONIC, &t_lcs_start);
    //fill up the rest of the matrix and return final score (element locate at the last line and collumn)
    mtype score = LCS(scoreMatrix, sizeA, sizeB, seqA, seqB);
    clock_gettime(CLOCK_MONOTONIC, &t_lcs_end);

	/* if you wish to see the entire score matrix,
	 for debug purposes, define DEBUGMATRIX. */
#ifdef DEBUGMATRIX
    printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
#endif

    printf("\nScore: %d\n", score);

    //free score matrix
    clock_gettime(CLOCK_MONOTONIC, &t_free);
    freeScoreMatrix(scoreMatrix, sizeB);
    free(seqA);
    free(seqB);

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    // time
    double t_total = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    double t_reading = (t_alloc.tv_sec - t_read.tv_sec) + (t_alloc.tv_nsec - t_read.tv_nsec) / 1e9;
    double t_alloc_init = (t_lcs_start.tv_sec - t_alloc.tv_sec) + (t_lcs_start.tv_nsec - t_alloc.tv_nsec) / 1e9;
    double t_lcs = (t_lcs_end.tv_sec - t_lcs_start.tv_sec) + (t_lcs_end.tv_nsec - t_lcs_start.tv_nsec) / 1e9;
    double t_cleanup = (t_end.tv_sec - t_free.tv_sec) + (t_end.tv_nsec - t_free.tv_nsec) / 1e9;

    printf("Total:  %.6f s\n", t_total);
    printf("Read:   %.6f s\n", t_reading);
    printf("Alocc:  %.6f s\n", t_alloc_init);
    printf("LCS:    %.6f s\n", t_lcs);
    printf("Free:   %.6f s\n", t_cleanup);

    return EXIT_SUCCESS;
}

