#!/bin/zsh
# PedNet (15 epochs) -> evaluation -> report.
cd /Users/maksymkhodakov/PycharmProjects/ImageRecognitionProject
echo "PedNet (15 epochs) started at $(date)" >> logs/queue.log
.venv/bin/python -u train_pednet.py --epochs 15 --batch 16 --workers 4 > logs/train_pednet.log 2>&1
echo "PedNet finished at $(date)" >> logs/queue.log
.venv/bin/python -u evaluate.py > logs/evaluate.log 2>&1 && echo "evaluate finished at $(date)" >> logs/queue.log
.venv/bin/python -u report/build_report.py > logs/report.log 2>&1 && echo "report built at $(date)" >> logs/queue.log
echo "ALL DONE at $(date)" >> logs/queue.log
