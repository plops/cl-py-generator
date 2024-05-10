NAME=$1

# VMj-3S1tku0 The spelled-out intro to neural networks and backpropagation: 1.5M views 1 year ago
# PaCmpygFfXo The spelled-out intro to language modeling: building makemore 584K views 1 year ago
# TCH_1BHY58I Makemore Part 2: MLP: 273K views 1 year ago  273k views
# P6sfmUTpUmc Makemore Part 3: Activations, BatchNorm      243k views
# q8SA3rM6ckI Makemore Part 4: Backprop                    169k views
# t3YJ5hKiMQ0 Makemore Part 5: WaveNet                     155k views
# kCc8FmEb1nY GPT from scratch                             4.2M views
# zduSFxRajkE GPT Tokenizer                                464k views
 
# Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh

LANG=english
ODIR=.
MODEL=~/src/whisper.cpp/models/ggml-large-v3.bin

# sudo apt install yt-dlp ffmpeg

yt-dlp -f 251 $NAME -o $NAME.webm

ffmpeg -y -i $NAME.webm -ar 16000 -ac 1 -c:a pcm_s16le $ODIR/output.wav

~/src/whisper.cpp/bgpu/bin/main -t 30 -l $LANG -m $MODEL -f $ODIR/output.wav -otxt -nt -of $LANG.txt -osrt $LANG.srt 2>&1 | tee $LANG_tee.txt

# GPU usage:
# | 65%   85C    P2             136W / 140W |   4918MiB / 16376Mi

echo -e 'Summarize the following video transcript as a bullet list. Prepend each bullet point with the starting timestamp. Do not show the end timestamps. Split the summary into sections and create section titles:\n```\n' > $NAME.md 
awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' $LANG.txt.srt >> $NAME.md
echo -e '```\nShow the english summary.' >> $NAME.md
