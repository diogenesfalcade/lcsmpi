#!/bin/bash
#SBATCH --job-name=lcs_mpi
#SBATCH --output=lcs_output_%j.out      # Arquivo de saída (stdout)
#SBATCH --error=lcs_error_%j.err        # Arquivo de erro (stderr)
#SBATCH --ntasks=4                      # Número total de processos MPI
#SBATCH --time=00:10:00                 # Tempo máximo
#SBATCH --partition=normal              # Fila/partição do SLURM (ajuste se necessário)
#SBATCH --nodes=2                       # Número de nós
#SBATCH --ntasks-per-node=2            # Processos MPI por nó (ajuste conforme o cluster)

module load mpi                        # Ajuste se necessário
mpirun ./lcs_mpi_exec
