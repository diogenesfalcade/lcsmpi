#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

// Tamanho do bloco (ajuste para otimizar o desempenho)
#define BLOCK_SIZE 16

// Macro para o máximo entre dois números
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// Macro para o mínimo entre dois números
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Função para ler o conteúdo de um arquivo para uma string (char*)
char* read_file_to_string(const char* filename, long* length) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Erro ao abrir o arquivo");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char*)malloc(*length + 1);
    if (buffer == NULL) {
        fprintf(stderr, "Falha na alocação de memória para o arquivo.\n");
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, *length, file);
    buffer[*length] = '\0';
    fclose(file);
    return buffer;
}

// Função para computar a tabela de PD para um único bloco em C
void compute_lcs_block(
    const char* s1_chunk, int m,
    const char* s2_chunk, int n,
    const int* top_boundary,
    const int* left_boundary,
    int* right_boundary,
    int* bottom_boundary)
{
    // Aloca a tabela de DP local para este bloco
    int* dp = (int*)malloc((m + 1) * (n + 1) * sizeof(int));

    // Inicializa as bordas da tabela de DP local
    for (int j = 0; j <= n; ++j) dp[0 * (n + 1) + j] = top_boundary[j];
    for (int i = 0; i <= m; ++i) dp[i * (n + 1) + 0] = left_boundary[i];

    // Preenche a tabela de DP local
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (s1_chunk[i - 1] == s2_chunk[j - 1]) {
                dp[i * (n + 1) + j] = dp[(i - 1) * (n + 1) + (j - 1)] + 1;
            } else {
                dp[i * (n + 1) + j] = MAX(dp[(i - 1) * (n + 1) + j], dp[i * (n + 1) + (j - 1)]);
            }
        }
    }

    // Extrai as bordas direita e inferior
    for (int i = 0; i <= m; ++i) right_boundary[i] = dp[i * (n + 1) + n];
    for (int j = 0; j <= n; ++j) bottom_boundary[j] = dp[m * (n + 1) + j];
    
    free(dp);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *s1 = NULL, *s2 = NULL;
    long M_long, N_long;
    int M, N;
    struct timespec t_start, t_end;

    if (rank == 0) {
        s1 = read_file_to_string("fileA.in", &M_long);
        s2 = read_file_to_string("fileB.in", &N_long);
        if (s1 == NULL || s2 == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        M = (int)M_long;
        N = (int)N_long;

?
    }
    
    // Transmite os tamanhos
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Aloca memória nas outras threads
    if (rank != 0) {
        s1 = (char*)malloc(M + 1);
        s2 = (char*)malloc(N + 1);
    }

    // Transmite as strings
    MPI_Bcast(s1, M + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(s2, N + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    int num_blocks_i = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_j = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Aloca memória para as bordas esquerdas locais
    int** local_left_boundaries = (int**)malloc((num_blocks_j + 1) * sizeof(int*));
    for(int j = 0; j <= num_blocks_j; ++j) {
        local_left_boundaries[j] = (int*)calloc(BLOCK_SIZE + 1, sizeof(int));
    }
    
    // Sincroniza todos os processos antes de iniciar o cronômetro
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t_start);
    }

    // Loop principal da computação (Wavefront)
    for (int diag = 0; diag < num_blocks_i + num_blocks_j - 1; ++diag) {
        for (int bi = 0; bi < num_blocks_i; ++bi) {
            int bj = diag - bi;
            if (bj >= 0 && bj < num_blocks_j) {
                if (bi % size == rank) {
                    int s1_start = bi * BLOCK_SIZE;
                    int s1_len = MIN(BLOCK_SIZE, M - s1_start);
                    int s2_start = bj * BLOCK_SIZE;
                    int s2_len = MIN(BLOCK_SIZE, N - s2_start);

                    int* top_boundary = (int*)malloc((s2_len + 1) * sizeof(int));
                    
                    if (bi > 0) {
                        MPI_Status status;
                        int source_rank = (rank - 1 + size) % size;
                        MPI_Recv(top_boundary, s2_len + 1, MPI_INT, source_rank, bj, MPI_COMM_WORLD, &status);
                    } else {
                        memset(top_boundary, 0, (s2_len + 1) * sizeof(int));
                    }

                    int* left_boundary = local_left_boundaries[bj];
                    int* right_boundary = (int*)malloc((s1_len + 1) * sizeof(int));
                    int* bottom_boundary = (int*)malloc((s2_len + 1) * sizeof(int));

                    compute_lcs_block(s1 + s1_start, s1_len, s2 + s2_start, s2_len, 
                                      top_boundary, left_boundary, right_boundary, bottom_boundary);

                    // Libera a borda antiga e armazena a nova
                    free(local_left_boundaries[bj + 1]);
                    local_left_boundaries[bj + 1] = right_boundary;

                    if (bi < num_blocks_i - 1) {
                        int dest_rank = (rank + 1) % size;
                        MPI_Send(bottom_boundary, s2_len + 1, MPI_INT, dest_rank, bj, MPI_COMM_WORLD);
                    }
                    
                    free(top_boundary);
                    free(bottom_boundary);
                }
            }
        }
    }

    // Sincroniza todos os processos antes de parar o cronômetro
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        
        int final_lcs_length;
        int last_block_owner_rank = (num_blocks_i - 1) % size;

        if (last_block_owner_rank == 0) {
            int last_s1_len = MIN(BLOCK_SIZE, M - (num_blocks_i - 1) * BLOCK_SIZE);
            final_lcs_length = local_left_boundaries[num_blocks_j][last_s1_len];
        } else {
            MPI_Status status;
            MPI_Recv(&final_lcs_length, 1, MPI_INT, last_block_owner_rank, 0, MPI_COMM_WORLD, &status);
        }

        double time_taken = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        
        printf("\n----------------------------------------\n");
        printf("Computação finalizada.\n");
        printf("Score (Comprimento da LCS): %d\n", final_lcs_length);
        printf("Tempo de execução: %f s\n", time_taken);
        printf("----------------------------------------\n");

        FILE *fp = fopen("resultados.txt", "a");
        if (fp == NULL) {
            perror("Erro ao abrir 'resultados.txt'");
        } else {
            fprintf(fp, "Score: %d, Time: %f s, Procs: %d\n", final_lcs_length, time_taken, size);
            fclose(fp);
            printf("Resultado salvo em 'resultados.txt'\n");
        }

    } else if (rank == (num_blocks_i - 1) % size) {
        // O processo que tem o resultado final o envia para o processo 0
        int last_s1_len = MIN(BLOCK_SIZE, M - (num_blocks_i - 1) * BLOCK_SIZE);
        int final_lcs_length = local_left_boundaries[num_blocks_j][last_s1_len];
        MPI_Send(&final_lcs_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Limpeza de memória
    free(s1);
    free(s2);
    for(int j = 0; j <= num_blocks_j; ++j) {
        free(local_left_boundaries[j]);
    }
    free(local_left_boundaries);

    MPI_Finalize();
    return 0;
}