#!/bin/bash
EXPERIMENT_PATH=$HOME/TCC/Capsule-Networks
for i in {1..5}; do
  echo -e "\nEXECUTION $i\n"
  source $HOME/torch/bin/activate
  python "$EXPERIMENT_PATH/main.py" --model capsnet_without --epochs 50 --patience 10 --batch_size 16 --routing_iterations 3
  mv $EXPERIMENT_PATH/metrics.csv $EXPERIMENT_PATH/results/metrics_run$i.csv
  mv $EXPERIMENT_PATH/final_loss.txt $EXPERIMENT_PATH/results/final_loss_run$i.txt
  mv $EXPERIMENT_PATH/capsnet_without_best_model.pth $EXPERIMENT_PATH/results/capsnet_without_best_model_run$i.pth
done
