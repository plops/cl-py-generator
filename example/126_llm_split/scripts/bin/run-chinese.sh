
# ffmpeg -i Codellama\ Deep\ Dive-Y0gYsq7tOnM.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav
time ./main -l chinese -t 12  -m ~/src/whisper.cpp/models/ggml-base.bin -f /home/martin/output.wav -otxt -nt -of chinese.txt   -osrt distance -ovtt distance_vtt -olrc distance_lrc 2>&1 | tee chinese_tee.txt
