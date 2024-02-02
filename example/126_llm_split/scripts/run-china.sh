NAME=QsUiAqVYOZQ
cd ~
ssh hetzner yt-dlp -f 251 $NAME -o $NAME.webm
scp hetzner:./$NAME.webm .
ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le ~/output.wav
cd /home/martin/src/whisper.cpp/b/bin
sh run-chinese.sh
mkdir $NAME
cp chinese* $NAME
cp $NAME/*.srt /dev/shm
cd ~/gpt
mkdir $NAME
cd $NAME
cp /dev/shm/ch*.srt .
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' *.srt > 1
# python3 -i /usr/local/bin/split.py 1
python3 /usr/local/bin/split_transcript.py --chunk_size 500 --prompt "Summarize the following video transcript as a bullet list. Prepend each bullet point with the starting timestamp. Don't show the end timestamps. Don't just write abstract summary but also give details, deduce opinion and conclusions from the words in the transcript. Write names in english (pinyin) transliteration as well as chinese characters. Also split the summary into sections and create section titles (a section title should convey the content and may not include a number). A title shall be written like so: *this is a title*. A bullet shall be written like so: - 01:32 text of bullet. Note that the timestamp is just at the beginning and not fat. " 1
cp ../sum .
