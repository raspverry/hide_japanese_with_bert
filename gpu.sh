
# azureはsecure bootを無効化しないといけない

# listで互換性あるnvidia driverを確認
sudo ubuntu-drivers list

# 選択したdriverをダウンロード
sudo ubuntu-drivers install nvidia-driver-535-server

sudo reboot

# driver check
sudo nvidia-smi

# nvcc -Versionでcuda driverがあるかチェック

sudo apt install -y gcc-11 g++-11
# gcc 11をalternative versionに設定
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11

# 現在セッションで使うcomplierの設定
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# cuda driverダウンロードして設置
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run --toolkit --samples --silent

echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc