echo "export LD_LIBRARY_PATH=/home/aistudio/cuda/cuda-9.2/lib64:\$LD_LIBRARY_PATH" >> .bashrc
echo "export CUDA_HOME=/home/aistudio/cuda/cuda-9.2/:\$CUDA_HOME" >> .bashrc
python3 -m pip install paddlepaddle-gpu==1.8.3.post97 -i https://mirror.baidu.com/pypi/simple
pip install -r requirements.txt
unzip data/data48884/data.zip
source .bashrc