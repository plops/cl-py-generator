NAME=P4DThUr7bsY
cd ~
ssh hetzner yt-dlp -f 251 $NAME -o $NAME.webm
scp hetzner:./$NAME.webm .
ffmpeg -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le ~/output.wav
cd /home/martin/src/whisper.cpp/b/bin
sh run-japanese.sh
mkdir $NAME
cp jap* $NAME
cp $NAME/*.srt /dev/shm
cd ~/gpt
mkdir $NAME
cd $NAME
cp /dev/shm/jap*.srt .
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' *.srt > 1
python3 -i /usr/local/bin/split.py 1
cp ../sum .
