# Create conda environment
conda create -n 3.10.12 python=3.10.12
conda activate 3.10.12
# Clone the repo
git clone https://github.com/haoheliu/AudioLDM-training-finetuning.git; cd ZSS_att1
# Install running environment
pip install poetry
poetry install
