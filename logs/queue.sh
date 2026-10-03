#!/bin/zsh
cd /Users/maksymkhodakov/PycharmProjects/ImageRecognitionProject
while kill -0 76393 2>/dev/null; do sleep 60; done
echo "YOLO finished at $(date)" >> logs/queue.log
.venv/bin/python -u train_pednet.py --epochs 25 --batch 16 --workers 4 > logs/train_pednet.log 2>&1
echo "PedNet finished at $(date)" >> logs/queue.log
.venv/bin/python -u evaluate.py > logs/evaluate.log 2>&1 && echo "evaluate finished at $(date)" >> logs/queue.log
.venv/bin/python -u report/build_report.py > logs/report.log 2>&1 && echo "report built at $(date)" >> logs/queue.log
echo "ALL DONE at $(date)" >> logs/queue.log
