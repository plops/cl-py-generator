# runs as many jobs as there are available vCPUs
# sudo dnf install parallel
mkdir data
parallel < 01_download_audio.sh 
