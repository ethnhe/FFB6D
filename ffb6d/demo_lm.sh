#!/bin/bash
cls='ape'
tst_mdl=train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar
python3 -m demo -dataset linemod -checkpoint $tst_mdl -cls $cls -show
