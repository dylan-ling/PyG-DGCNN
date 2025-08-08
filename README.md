# DGCNN / PyTorch-Geometric Workflow

Repository to implement a DGCNN-based point cloud classifier on ModelNet40 using Pytorch Geometric

- Python 3.8+, CUDA 11.7+
- PyTorch, PyGeometric, NumPy

How to Run:

Pure Edge Conv:

python pyg_main.py --exp_name=dgcnn_2048 --model=dgcnn --num_points=2048 --k=20 --use_sgd


PyGeometric:

python newpyg_main.py --exp_name=dgcnn_2048 --model=dgcnn --num_points=2048 --k=20 --use_sgd

