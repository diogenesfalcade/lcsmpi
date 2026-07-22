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
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

typedef unsigned short mtype;

char* read_seq(const char* fname) {
    FILE* f = fopen(fname, "rt");
    if (!f) {
        perror("Erro ao abrir arquivo");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char* seq = (char*)calloc(size + 1, sizeof(char));
    if (fread(seq, sizeof(char), size, f) != size) {
        perror("Erro na leitura do arquivo");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

int block_owner(int bx, int by, int size) {
    static int global_id = 0;
    return (bx + by + global_id++) % size;
}

int LCS_mpi(char* seqA, char* seqB, int sizeA, int sizeB, int rank, int size) {
    int nBX = (sizeA + TILE_SIZE - 1) / TILE_SIZE;
    int nBY = (sizeB + TILE_SIZE - 1) / TILE_SIZE;
    mtype** score = allocateMatrix(sizeB + 1, sizeA + 1);

    // Percorre as diagonais de blocos
    for (int d = 0; d < nBX + nBY - 1; d++) {
        int global_block_id = 0; // reinicia ID por diagonal

        for (int by = max(0, d - nBX + 1); by <= min(d, nBY - 1); by++) {
            int bx = d - by;

            if ((global_block_id % size) != rank) {
                global_block_id++;
                continue;
            }

            // Coordenadas do bloco atual
            int rs = by * TILE_SIZE + 1;
            int re = min((by + 1) * TILE_SIZE, sizeB);
            int cs = bx * TILE_SIZE + 1;
            int ce = min((bx + 1) * TILE_SIZE, sizeA);

            // Donos dos blocos de fronteira
            int upper_rank = ((bx + (by - 1)) >= 0) ? ((global_block_id - nBX) % size) : -1;
            int left_rank  = ((bx - 1 + by) >= 0) ? ((global_block_id - 1) % size) : -1;

            // Recebe da borda superior
            if (by > 0 && upper_rank != rank)
                MPI_Recv(score[rs - 1] + cs, ce - cs + 1, MPI_UNSIGNED_SHORT, upper_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Recebe da borda esquerda
            if (bx > 0 && left_rank != rank)
                for (int i = rs; i <= re; i++)
                    MPI_Recv(&score[i][cs - 1], 1, MPI_UNSIGNED_SHORT, left_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Cálculo do bloco
            computeBlock(score, seqA, seqB, rs, cs, re, ce);

            // Envia borda inferior
            if (by < nBY - 1) {
                int down_rank = ((global_block_id + nBX) % size);
                if (down_rank != rank)
                    MPI_Send(score[re] + cs, ce - cs + 1, MPI_UNSIGNED_SHORT, down_rank, 0, MPI_COMM_WORLD);
            }

            // Envia borda direita
            if (bx < nBX - 1) {
                int right_rank = ((global_block_id + 1) % size);
                if (right_rank != rank)
                    for (int i = rs; i <= re; i++)
                        MPI_Send(&score[i][ce], 1, MPI_UNSIGNED_SHORT, right_rank, 1, MPI_COMM_WORLD);
            }

            global_block_id++;
        }

        MPI_Barrier(MPI_COMM_WORLD); // sincroniza por diagonal
    }

    int score_final = score[sizeB][sizeA];
    freeMatrix(score, sizeB + 1);
    return score_final;
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

    if (rank == 0)
        clock_gettime(CLOCK_MONOTONIC, &t_start);

    int score = LCS_mpi(seqA, seqB, sizeA, sizeB, rank, size);

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        printf("Score: %d\n", score);
        printf("Elapsed time: %.6f s\n", elapsed);

        FILE* fout = fopen("resultados.txt", "a");
        if (fout) {
            fprintf(fout, "Score: %d, Time: %.6f s, Procs: %d\n", score, elapsed, size);
            fclose(fout);
        }
    }

    free(seqA);
    free(seqB);
    MPI_Finalize();
    return 0;
}
