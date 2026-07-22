#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// #define DEBUGMATRIX

typedef unsigned short mtype;

// tile size
#define TILE_SIZE 256

// Structure to hold block information
typedef struct {
    int bx, by;             // Block coordinates
    int row_start, row_end; // Row bounds
    int col_start, col_end; // Column bounds
} BlockInfo;

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

void processBlock(mtype **scoreMatrix, BlockInfo block, char *seqA, char *seqB) {
    // standard LCS update inside the block
    for (int i = block.row_start; i <= block.row_end; i++) {
        for (int j = block.col_start; j <= block.col_end; j++) {
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

int LCS(mtype **scoreMatrix, int sizeA, int sizeB, char *seqA, char *seqB, int rank, int num_procs) {
    // calculate how many blocks (tiles) are needed in both dimensions
    int nBlocksX = (sizeA + TILE_SIZE - 1) / TILE_SIZE;
    int nBlocksY = (sizeB + TILE_SIZE - 1) / TILE_SIZE;
    int total_blocks = nBlocksX * nBlocksY;
    
    // Each process will work on its assigned blocks in anti-diagonal order
    for (int diag = 0; diag < nBlocksX + nBlocksY - 1; diag++) {
        for (int bx = 0; bx < nBlocksX; bx++) {
            int by = diag - bx;
            if (by >= 0 && by < nBlocksY) {
                // Check if this block should be processed by current process
                int block_idx = by * nBlocksX + bx;
                if (block_idx % num_procs == rank) {
                    BlockInfo block;
                    block.bx = bx;
                    block.by = by;
                    
                    // define row and column bounds for this block
                    block.row_start = by * TILE_SIZE + 1;
                    block.row_end   = (by + 1) * TILE_SIZE;
                    if (block.row_end > sizeB) block.row_end = sizeB;

                    block.col_start = bx * TILE_SIZE + 1;
                    block.col_end   = (bx + 1) * TILE_SIZE;
                    if (block.col_end > sizeA) block.col_end = sizeA;
                    
                    // Process the block
                    processBlock(scoreMatrix, block, seqA, seqB);
                    
                    // Send the right and bottom borders to neighboring blocks
                    // Send right border to right neighbor (if exists)
                    if (bx < nBlocksX - 1) {
                        // Prepare right border data (last column of current block)
                        mtype *right_border = (mtype *)malloc((block.row_end - block.row_start + 2) * sizeof(mtype));
                        for (int i = block.row_start - 1; i <= block.row_end; i++) {
                            right_border[i - (block.row_start - 1)] = scoreMatrix[i][block.col_end];
                        }
                        
                        // Determine target rank for right neighbor block
                        int right_block_idx = block_idx + 1;
                        int target_rank = right_block_idx % num_procs;
                        
                        MPI_Send(right_border, block.row_end - block.row_start + 2, MPI_UNSIGNED_SHORT, 
                                target_rank, block_idx, MPI_COMM_WORLD);
                        free(right_border);
                    }
                    
                    // Send bottom border to bottom neighbor (if exists)
                    if (by < nBlocksY - 1) {
                        // Prepare bottom border data (last row of current block)
                        mtype *bottom_border = (mtype *)malloc((block.col_end - block.col_start + 2) * sizeof(mtype));
                        for (int j = block.col_start - 1; j <= block.col_end; j++) {
                            bottom_border[j - (block.col_start - 1)] = scoreMatrix[block.row_end][j];
                        }
                        
                        // Determine target rank for bottom neighbor block
                        int bottom_block_idx = block_idx + nBlocksX;
                        int target_rank = bottom_block_idx % num_procs;
                        
                        MPI_Send(bottom_border, block.col_end - block.col_start + 2, MPI_UNSIGNED_SHORT, 
                                target_rank, block_idx, MPI_COMM_WORLD);
                        free(bottom_border);
                    }
                }
            }
        }
        
        // After processing all blocks in this diagonal, receive border data if needed
        for (int bx = 0; bx < nBlocksX; bx++) {
            int by = diag - bx;
            if (by >= 0 && by < nBlocksY) {
                int block_idx = by * nBlocksX + bx;
                if (block_idx % num_procs == rank) {
                    // Receive left border from left neighbor (if exists)
                    if (bx > 0) {
                        int left_block_idx = block_idx - 1;
                        int source_rank = left_block_idx % num_procs;
                        
                        BlockInfo block;
                        block.bx = bx;
                        block.by = by;
                        block.row_start = by * TILE_SIZE + 1;
                        block.row_end   = (by + 1) * TILE_SIZE;
                        if (block.row_end > sizeB) block.row_end = sizeB;
                        block.col_start = bx * TILE_SIZE + 1;
                        block.col_end   = (bx + 1) * TILE_SIZE;
                        if (block.col_end > sizeA) block.col_end = sizeA;
                        
                        mtype *left_border = (mtype *)malloc((block.row_end - block.row_start + 2) * sizeof(mtype));
                        MPI_Status status;
                        MPI_Recv(left_border, block.row_end - block.row_start + 2, MPI_UNSIGNED_SHORT, 
                                source_rank, left_block_idx, MPI_COMM_WORLD, &status);
                        
                        // Update the left border in our matrix
                        for (int i = block.row_start - 1; i <= block.row_end; i++) {
                            scoreMatrix[i][block.col_start - 1] = left_border[i - (block.row_start - 1)];
                        }
                        free(left_border);
                    }
                    
                    // Receive top border from top neighbor (if exists)
                    if (by > 0) {
                        int top_block_idx = block_idx - nBlocksX;
                        int source_rank = top_block_idx % num_procs;
                        
                        BlockInfo block;
                        block.bx = bx;
                        block.by = by;
                        block.row_start = by * TILE_SIZE + 1;
                        block.row_end   = (by + 1) * TILE_SIZE;
                        if (block.row_end > sizeB) block.row_end = sizeB;
                        block.col_start = bx * TILE_SIZE + 1;
                        block.col_end   = (bx + 1) * TILE_SIZE;
                        if (block.col_end > sizeA) block.col_end = sizeA;
                        
                        mtype *top_border = (mtype *)malloc((block.col_end - block.col_start + 2) * sizeof(mtype));
                        MPI_Status status;
                        MPI_Recv(top_border, block.col_end - block.col_start + 2, MPI_UNSIGNED_SHORT, 
                                source_rank, top_block_idx, MPI_COMM_WORLD, &status);
                        
                        // Update the top border in our matrix
                        for (int j = block.col_start - 1; j <= block.col_end; j++) {
                            scoreMatrix[block.row_start - 1][j] = top_border[j - (block.col_start - 1)];
                        }
                        free(top_border);
                    }
                }
            }
        }
        
        // Synchronize all processes before moving to next diagonal
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // After all blocks are processed, gather the final result at root
    if (rank == 0) {
        // Receive matrix parts from other processes
        for (int src = 1; src < num_procs; src++) {
            for (int by = 0; by < nBlocksY; by++) {
                for (int bx = 0; bx < nBlocksX; bx++) {
                    int block_idx = by * nBlocksX + bx;
                    if (block_idx % num_procs == src) {
                        BlockInfo block;
                        block.bx = bx;
                        block.by = by;
                        block.row_start = by * TILE_SIZE + 1;
                        block.row_end   = (by + 1) * TILE_SIZE;
                        if (block.row_end > sizeB) block.row_end = sizeB;
                        block.col_start = bx * TILE_SIZE + 1;
                        block.col_end   = (bx + 1) * TILE_SIZE;
                        if (block.col_end > sizeA) block.col_end = sizeA;
                        
                        // Receive the block data
                        for (int i = block.row_start; i <= block.row_end; i++) {
                            MPI_Recv(&scoreMatrix[i][block.col_start], 
                                    block.col_end - block.col_start + 1, 
                                    MPI_UNSIGNED_SHORT, src, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                    }
                }
            }
        }
    } else {
        // Send my blocks to root
        for (int by = 0; by < nBlocksY; by++) {
            for (int bx = 0; bx < nBlocksX; bx++) {
                int block_idx = by * nBlocksX + bx;
                if (block_idx % num_procs == rank) {
                    BlockInfo block;
                    block.bx = bx;
                    block.by = by;
                    block.row_start = by * TILE_SIZE + 1;
                    block.row_end   = (by + 1) * TILE_SIZE;
                    if (block.row_end > sizeB) block.row_end = sizeB;
                    block.col_start = bx * TILE_SIZE + 1;
                    block.col_end   = (bx + 1) * TILE_SIZE;
                    if (block.col_end > sizeA) block.col_end = sizeA;
                    
                    // Send the block data row by row
                    for (int i = block.row_start; i <= block.row_end; i++) {
                        MPI_Send(&scoreMatrix[i][block.col_start], 
                                block.col_end - block.col_start + 1, 
                                MPI_UNSIGNED_SHORT, 0, i, MPI_COMM_WORLD);
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
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // measure time
    struct timespec t_start, t_read, t_alloc, t_lcs_start, t_lcs_end, t_free, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // sequence pointers for both sequences
    char *seqA, *seqB;

    // sizes of both sequences
    int sizeA, sizeB;

    clock_gettime(CLOCK_MONOTONIC, &t_read);
    //read both sequences (only root needs this)
    if (rank == 0) {
        seqA = read_seq("fileA.in");
        seqB = read_seq("fileB.in");
        sizeA = strlen(seqA);
        sizeB = strlen(seqB);
    }
    
    // Broadcast sizes to all processes
    MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate sequences in other processes
    if (rank != 0) {
        seqA = (char *)malloc((sizeA + 1) * sizeof(char));
        seqB = (char *)malloc((sizeB + 1) * sizeof(char));
    }
    
    // Broadcast sequences to all processes
    MPI_Bcast(seqA, sizeA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seqB, sizeB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // allocate LCS score matrix
    clock_gettime(CLOCK_MONOTONIC, &t_alloc);
    mtype **scoreMatrix = allocateScoreMatrix(sizeA, sizeB);

    //initialize LCS score matrix
    initScoreMatrix(scoreMatrix, sizeA, sizeB);

    clock_gettime(CLOCK_MONOTONIC, &t_lcs_start);
    //fill up the rest of the matrix and return final score (element locate at the last line and collumn)
    mtype score = LCS(scoreMatrix, sizeA, sizeB, seqA, seqB, rank, num_procs);
    clock_gettime(CLOCK_MONOTONIC, &t_lcs_end);

    if (rank == 0) {
        /* if you wish to see the entire score matrix,
         for debug purposes, define DEBUGMATRIX. */
    #ifdef DEBUGMATRIX
        printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
    #endif

        printf("\nScore: %d\n", score);
    }

    //free score matrix
    clock_gettime(CLOCK_MONOTONIC, &t_free);
    freeScoreMatrix(scoreMatrix, sizeB);
    free(seqA);
    free(seqB);

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    if (rank == 0) {
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
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}