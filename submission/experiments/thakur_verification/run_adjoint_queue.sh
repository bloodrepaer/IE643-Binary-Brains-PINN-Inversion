#!/bin/bash
cd /home/claude/thakur
for args in "--case I --n-spatial 25" "--case I --n-spatial 2500" "--case II --n-spatial 2500" "--case III --n-spatial 2500" "--case IV --n-spatial 2500" "--case I --n-spatial 2500 --noise 0.01" "--case IV --n-spatial 2500 --noise 0.01"; do
  echo "== $args"; python classical.py adjoint $args 2>&1 | grep -v -i warning | tail -1 | cut -c1-400
done
