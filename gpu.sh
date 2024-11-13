# remove previous package
sudo apt-get purge -y "*nvidia*" "*cuda*"
sudo apt-get autoremove -y
sudo apt-get clean

# nouveau driver inactive
echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

# CUDA setting
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# system update
sudo apt-get update

# NVIDIA driver & cuda install
sudo apt-get -y install cuda-drivers-535
sudo apt-get -y install cuda

# old key and key storage remove (since they are deprecated)
sudo rm /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list
sudo rm /etc/apt/trusted.gpg.d/cuda-ubuntu2204-keyring.gpg || true

# new way CUDA install
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt install nvidia-cuda-toolkit




# # 1. remove previous install
# sudo apt-get purge -y "*nvidia*" "*cuda*"
# sudo apt-get autoremove -y
# sudo apt-get clean

# # 2. CUDA 11.7 storage setting (GiNZA recommended)
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update

# # 3.certain version install 
# sudo apt-get install -y cuda-11-7

# # 4. set env path
# echo 'export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
# source ~/.bashrc