
# ffmpeg -i Codellama\ Deep\ Dive-Y0gYsq7tOnM.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav
 -m ~/src/whisper.cpp/models/ggml-base.bin 
time ./main -l japanese -t 10 -tr \
     -m ~/src/whisper.cpp/models/ggml-large-v3.bin \
     -f /home/martin/output.wav -otxt -nt -of jp.txt -osrt jp_distance -ovtt jp_distance_vtt -olrc jp_distance_lrc  | tee jp_tee.txt
