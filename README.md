Project
OpenMP Numerical Integration (Trapezoid & Simpson) with performance studies (Parts A–E).
Prerequisites

Linux/macOS or GitHub Codespaces / WSL
gcc with OpenMP: sudo apt update && sudo apt install -y build-essential
Python 3 + libs for plots (optional):

    * python3 -m venv .venv && source .venv/bin/activate
    * pip install matplotlib pandas numpy

Build
    gcc -O3 -march=native -std=c11 -fopenmp myfile.c -o myprogram -lm

Quick Run
    # Trapezoid, 1 thread, sanity check (≈2.0 result)
    ./myfile --rule trap --func sin --a 0 --b 3.141592653589793 \
                --n 1048576 --threads 1 --work 0 --schedule static,1

Parts (how to reproduce)
Part A / Part E — Accuracy & Error Slopes
mkdir -p partE
: > partE/accuracy_trap.json
for k in $(seq 6 14); do
  n=$((1<<k))
  ./myprogram --rule trap --func sin --a 0 --b 3.141592653589793 \
              --n $n --threads 1 --work 0 --repeat 3 --warmups 1 \
    | tee -a partE/accuracy_trap.json
done

: > partE/accuracy_simp.json
for k in $(seq 6 14); do
  n=$((1<<k))
  ./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
              --n $n --threads 1 --work 0 --repeat 3 --warmups 1 \
    | tee -a partE/accuracy_simp.json
done

# Plot (produces partE/error_vs_h.png; prints fitted slopes)
python3 plot_error.py \
  --files partE/accuracy_trap.json partE/accuracy_simp.json \
  --labels trap simp --true 2.0 \
  --title "Error vs h (sin on [0,π])" \
  --out partE/error_vs_h.png

  Part B — OpenMP Parallelization (demo)
./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
            --n $((1<<24)) --threads 4 --work 0 --schedule static,1 \
            --repeat 5 --warmups 2

Part C — Compute vs Memory; Reduction vs Padded
# Compute-bound sweep (small K for Codespaces)
mkdir -p partC
: > partC/compute.json
N=$((1<<21))
for K in 0 10 100; do
  ./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
              --n $N --threads 4 --work $K --schedule static,1 \
              --repeat 3 --warmups 1 --accum reduction \
    | tee -a partC/compute.json
done
python3 plot_partC_compute.py \
  --files partC/compute.json --metric median_s \
  --title "Runtime vs K (n=2^21, p=4)" \
  --out partC/runtime_vs_K.png

# False sharing (reduction vs padded)
: > partC/false_sharing.json
N=$((1<<23))
for ACC in reduction padded; do
  ./myprogram --rule trap --func sin --a 0 --b 3.141592653589793 \
              --n $N --threads 4 --work 0 --schedule static,1 \
              --repeat 3 --warmups 1 --accum $ACC \
    | tee -a partC/false_sharing.json
done
python3 plot_partC_accum.py \
  --files partC/false_sharing.json --metric median_s \
  --title "Reduction vs Padded (K=0, n=2^23, p=4)" \
  --out partC/accum_compare.png

  Part D — Strong & Weak Scaling (K=0 and K>0)
mkdir -p partD
# Strong (fixed N)
: > partD/strong_k0.json
N=$((1<<21))
for p in 1 2 4; do
  OMP_NUM_THREADS=$p ./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
                                 --n $N --threads $p --work 0 --schedule static,1 \
                                 --repeat 5 --warmups 2 | tee -a partD/strong_k0.json
done
: > partD/strong_k50.json
for p in 1 2 4; do
  OMP_NUM_THREADS=$p ./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
                                 --n $N --threads $p --work 50 --schedule static,1 \
                                 --repeat 5 --warmups 2 | tee -a partD/strong_k50.json
done

# Weak (N(p)=p*N1)
: > partD/weak_k0.json
N1=$((1<<22))
for p in 1 2 4; do
  N=$((N1*p))
  OMP_NUM_THREADS=$p ./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
                                 --n $N --threads $p --work 0 --schedule static,1 \
                                 --repeat 5 --warmups 2 | tee -a partD/weak_k0.json
done
: > partD/weak_k50.json
for p in 1 2 4; do
  N=$((N1*p))
  OMP_NUM_THREADS=$p ./myprogram --rule simp --func sin --a 0 --b 3.141592653589793 \
                                 --n $N --threads $p --work 50 --schedule static,1 \
                                 --repeat 5 --warmups 2 | tee -a partD/weak_k50.json
done

Reproducibility

Use warmups + repeats; report median.
Record environment variables (examples):

OMP_NUM_THREADS, OMP_PROC_BIND=spread, OMP_PLACES=cores


Note CPU/OS/compiler (gcc --version, /proc/cpuinfo).

Files

myfile.c (C/OpenMP program)
plot_error.py (error vs h)
plot_partC_compute.py (runtime vs K)
plot_partC_accum.py (reduction vs padded)
plot_weak_scaling.py (measured vs Gustafson)