NAME=S18irSR1tZw

# Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh

LANG=hindi 
ODIR=.
MODEL=~/src/whisper.cpp/models/ggml-large-v3.bin

# sudo apt install yt-dlp ffmpeg

yt-dlp -f 251 $NAME -o $NAME.webm

ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le $ODIR/output.wav

~/src/whisper.cpp/bgpu/bin/main -t 30 -l $LANG -m $MODEL -f $ODIR/output.wav -otxt -nt -of $LANG.txt -osrt $LANG.srt 2>&1 | tee $LANG_tee.txt

echo -e 'Summarize the following video transcript as a bullet list. Prepend each bullet point with the starting timestamp. Do not show the end timestamps. Split the summary into sections and create section titles:\n```\n' > $NAME.md 
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' $LANG.txt.srt >> $NAME.md
echo -e '```\nShow the english summary.' >> $NAME.md
