
# ffmpeg -i Codellama\ Deep\ Dive-Y0gYsq7tOnM.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav
time ./main -l english -t 8  -m ~/src/whisper.cpp/models/ggml-base.bin -f /home/martin/output.wav -otxt -nt -of en.txt -osrt en_distance -ovtt en_distance_vtt -olrc en_distance_lrc  2>&1 | tee en_tee.txt
