#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define TILE_SIZE 1024

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

typedef unsigned short mtype;

char* read_seq(const char* fname) {
    FILE* f = fopen(fname, "rt");
    if (!f) {
        perror("File open failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char* seq = (char*)calloc(size + 1, sizeof(char));
    fread(seq, sizeof(char), size, f);
    fclose(f);
    return seq;
}

mtype** allocateMatrix(int rows, int cols) {
    mtype** mat = malloc(rows * sizeof(mtype*));
    for (int i = 0; i < rows; i++)
        mat[i] = calloc(cols, sizeof(mtype));
    return mat;
}

void freeMatrix(mtype** mat, int rows) {
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}

void computeBlock(mtype** score, char* seqA, char* seqB,
                  int row_start, int col_start, int row_end, int col_end) {
    for (int i = row_start; i <= row_end; i++) {
        for (int j = col_start; j <= col_end; j++) {
            if (seqA[j - 1] == seqB[i - 1])
                score[i][j] = score[i - 1][j - 1] + 1;
            else
                score[i][j] = max(score[i - 1][j], score[i][j - 1]);
        }
    }
}

int LCS_mpi(char* seqA, char* seqB, int sizeA, int sizeB, int rank, int size, int* final_score_out) {
    int nBX = (sizeA + TILE_SIZE - 1) / TILE_SIZE;
    int nBY = (sizeB + TILE_SIZE - 1) / TILE_SIZE;

    mtype** score = allocateMatrix(sizeB + 1, sizeA + 1);

    for (int d = 0; d < nBX + nBY - 1; d++) {
        for (int by = max(0, d - nBX + 1); by <= (d < nBY ? d : nBY - 1); by++) {
            int bx = d - by;

            if ((by + bx) % size != rank) continue;

            int rs = by * TILE_SIZE + 1;
            int re = (by + 1) * TILE_SIZE;
            if (re > sizeB) re = sizeB;

            int cs = bx * TILE_SIZE + 1;
            int ce = (bx + 1) * TILE_SIZE;
            if (ce > sizeA) ce = sizeA;

            // Receber bordas
            if (by > 0) MPI_Recv(score[rs - 1] + cs, ce - cs + 1, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (bx > 0)
                for (int i = rs; i <= re; i++)
                    MPI_Recv(&score[i][cs - 1], 1, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            computeBlock(score, seqA, seqB, rs, cs, re, ce);

            // Enviar bordas
            int next_rank = (by + bx + 1) % size;
            if (by < nBY - 1)
                MPI_Send(score[re] + cs, ce - cs + 1, MPI_UNSIGNED_SHORT, next_rank, 0, MPI_COMM_WORLD);
            if (bx < nBX - 1)
                for (int i = rs; i <= re; i++)
                    MPI_Send(&score[i][ce], 1, MPI_UNSIGNED_SHORT, next_rank, 1, MPI_COMM_WORLD);

            // Se for o bloco final, armazene o score
            if (by == nBY - 1 && bx == nBX - 1) {
                *final_score_out = score[sizeB][sizeA];
                if (rank != 0)
                    MPI_Send(final_score_out, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    freeMatrix(score, sizeB + 1);
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    struct timespec t_start, t_end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *seqA = NULL, *seqB = NULL;
    int sizeA = 0, sizeB = 0;

    if (rank == 0) {
        seqA = read_seq("fileA.in");
        seqB = read_seq("fileB.in");
        sizeA = strlen(seqA);
        sizeB = strlen(seqB);
    }

    MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        seqA = malloc(sizeA + 1);
        seqB = malloc(sizeB + 1);
    }

    MPI_Bcast(seqA, sizeA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seqB, sizeB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    int final_score = 0;

    if (rank == 0)
        clock_gettime(CLOCK_MONOTONIC, &t_start);

    LCS_mpi(seqA, seqB, sizeA, sizeB, rank, size, &final_score);

    if (rank == 0) {
        // Recebe o score do último processo, se necessário
        int nBX = (sizeA + TILE_SIZE - 1) / TILE_SIZE;
        int nBY = (sizeB + TILE_SIZE - 1) / TILE_SIZE;
        int last_rank = (nBX - 1 + nBY - 1) % size;
        if (last_rank != 0)
            MPI_Recv(&final_score, 1, MPI_INT, last_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

        // printf("Score: %d\n", final_score);
        // printf("Elapsed time: %.6f s\n", elapsed);

        // Grava no arquivo resultados.txt
        FILE* fout = fopen("resultados.txt", "a");
        if (fout) {
            fprintf(fout, "Score: %d, Time: %.6f s, Procs: %d\n", final_score, elapsed, size);
            fclose(fout);
        } else {
            perror("Erro ao abrir resultados.txt");
        }
    }

    free(seqA);
    free(seqB);
    MPI_Finalize();
    return 0;
}
