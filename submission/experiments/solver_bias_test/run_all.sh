#!/bin/bash
cd /home/claude/solverbias
KEYS="C_inclusion__Fo0.1__uniform8__n50__random__eta0.01 C_inclusion__Fo0.1__uniform8__n5__random__eta0.01 B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05"
for D in data_rk4_129 data_cn_129 data_fe_129 data_cn_65_crime; do
  python adjoint_inversion.py --data $D --keys $KEYS --out res_adj_$D > log_adj_$D.txt 2>&1
  echo "adjoint done $D"
done
python adjoint_inversion.py --data data_rk4_129 --keys $KEYS --n-steps 50 --out res_adj_rk4_coarseCN > log_adj_coarse.txt 2>&1; echo "adjoint coarse done"
for D in data_rk4_129 data_cn_129 data_fe_129 data_cn_65_crime; do
  for K in $KEYS; do
    if [[ $K == B_* ]]; then R="--reg-type tikhonov --lambda-reg 2e-4"; else R="--reg-type tv --lambda-reg 3e-4"; fi
    python pinn_inversion.py --single-run --data-root $D --output-dir res_pinn_$D --exp-id $K $R --epochs 1500 > log_pinn_${D}_$K.txt 2>&1
    echo "pinn done $D $K"
  done
done
