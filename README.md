'''
# Create conda environment
conda create -n 3.10.12 python=3.10.12
conda activate 3.10.12

# Clone the repo
git clone https://github.com/wltschmrz/ZSS_att1.git; cd ZSS_att1

# Install running environment
pip install poetry
poetry install
pip install rp

# Download dataset & ckpts
cd data/
wget -c -O dataset.tar "https://zenodo.org/records/14342967/files/dataset.tar?download=1" && tar -xf dataset.tar
wget -c -O checkpoints.tar "https://zenodo.org/records/14342967/files/checkpoints.tar?download=1" && tar -xf checkpoints.tar
cd checkpoints/; wget -c "https://zenodo.org/records/7884686/files/audioldm-s-full" && mv "audioldm-s-full" "audioldm-s-full.ckpt"
cd ../../
'''
