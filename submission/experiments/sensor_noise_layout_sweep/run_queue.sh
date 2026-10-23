#!/bin/bash
# resumable: $1 = adjoint | pinn
cd /home/claude/v2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for K in $(cat /home/claude/sweep/keys.txt); do
  if [ "$1" == "adjoint" ]; then
    [ -f /home/claude/sweep/adj/$K/metrics.json ] && continue
    R=tikhonov; [[ $K == C_* ]] && R=tv
    python adjoint_inversion.py --keys $K --reg $R --out /home/claude/sweep/adj > /home/claude/sweep/log_adj_$K.txt 2>&1
  else
    [ -f /home/claude/sweep/pinn/$K/metrics.json ] && continue
    if [[ $K == B_* ]]; then R="--reg-type tikhonov --lambda-reg 2e-4"; else R="--reg-type tv --lambda-reg 3e-4"; fi
    python pinn_inversion.py --single-run --data-root /home/claude/v2 --output-dir /home/claude/sweep/pinn --exp-id $K $R --epochs 1500 > /home/claude/sweep/log_pinn_$K.txt 2>&1
  fi
  echo "$1 done $K"
done
echo "$1 QUEUE COMPLETE"
