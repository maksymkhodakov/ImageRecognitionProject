#!/bin/zsh
# Wait for YOLO training to finish, then train PedNet (sharing one GPU is too slow on M1 16GB).
cd /Users/maksymkhodakov/PycharmProjects/ImageRecognitionProject
while kill -0 74270 2>/dev/null; do sleep 60; done
echo "YOLO finished at $(date)" >> logs/queue.log
.venv/bin/python -u train_pednet.py --epochs 25 --batch 16 --workers 4 > logs/train_pednet.log 2>&1
echo "PedNet finished at $(date)" >> logs/queue.log
