rsync -av --progress *.ipynb calib2/*.nc slim:/dev/shm

# tunnel to slim
#     ssh slim -L 8888:localhost:8888 
# on slim
#     tmux
#     export WORKON_HOME=$HOME/.virtualenvs;export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3;source /usr/local/bin/virtualenvwrapper.sh;mkvirtualenv cv -p python3;jupyter notebook --no-browser
