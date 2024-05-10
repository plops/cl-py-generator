NAME=_pq6Rp8W6bo
ODIR=.
MODEL=~/src/whisper.cpp/models/ggml-large-v3.bin
# sudo apt install yt-dlp ffmpeg

yt-dlp -f 251 $NAME -o $NAME.webm

ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le $ODIR/output.wav

~/src/whisper.cpp/bgpu/bin/main -l chinese -m $MODEL -f $ODIR/output.wav -otxt -nt -of chinese.txt   -osrt chinese.srt -ovtt chinese.vtt -olrc chinese.lrc 2>&1 | tee chinese_tee.txt
