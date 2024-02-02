NAME=5HXMHdhQL_8
cd ~
ssh hetzner yt-dlp -f 251 $NAME -o $NAME.webm
scp hetzner:./$NAME.webm .
ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le ~/output.wav
cd /home/martin/src/whisper.cpp/b/bin
sh run-czech.sh
mkdir $NAME
cp cz* $NAME
cp $NAME/*.srt /dev/shm
cd ~/gpt
mkdir $NAME
cd $NAME
cp /dev/shm/*.srt .
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print substr($1, 4, 5)} NF==1' *.srt > 1
python3 /usr/local/bin/split_transcript.py --chunk_size 2400 --prompt "Summarize the following video transcript as a bullet list." 1
cp ../sum .
