NAME=s4BP5ZJJMZ8
cd ~
ssh hetzner yt-dlp -f 251 $NAME -o $NAME.webm
scp hetzner:./$NAME.webm .
ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le ~/output.wav
cd /home/martin/src/whisper.cpp/b/bin
sh run-en.sh
mkdir $NAME
cp en* $NAME
cp $NAME/*.srt /dev/shm
cd ~/gpt
mkdir $NAME
cd "$NAME"
cp /dev/shm/en*.srt .
#awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print substr($1, 4, 5)} NF==1' *.srt > 1
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' *.srt > 1
cp ../sum .
python3 /usr/local/bin/split_transcript.py --chunk_size 2400 --prompt "Summarize the following video transcript as a bullet list. Prepend each bullet point with starting timestamp. Don't show the ending timestamp. Also split the summary into sections and create section titles (a section title should convey the content and may not include a number). A title shall be written like so: *this is a title*. A bullet shall be written like so: - 01:32 text of bullet. Note that the timestamp is just at the beginning and not fat. " 1
