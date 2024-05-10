NAME=_pq6Rp8W6bo
LANG=chinese
ODIR=.
MODEL=~/src/whisper.cpp/models/ggml-large-v3.bin

# sudo apt install yt-dlp ffmpeg

yt-dlp -f 251 $NAME -o $NAME.webm

ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le $ODIR/output.wav

~/src/whisper.cpp/bgpu/bin/main -l $LANG -m $MODEL -f $ODIR/output.wav -otxt -nt -of $LANG.txt -osrt $LANG.srt 2>&1 | tee $LANG_tee.txt

echo -e 'Summarize the following video transcript as a bullet list. Prepend each bullet point with the starting timestamp. Do not show the end timestamps. Write names in english (pinyin) transliteration as well as chinese characters. Also split the summary into sections and create section titles:\n```\n' > $NAME.md 
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' *.srt >> $NAME.md
echo -e '```\nShow the english summary.' >> $NAME.md
