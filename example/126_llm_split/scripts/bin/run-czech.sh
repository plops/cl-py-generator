
# ffmpeg -i Codellama\ Deep\ Dive-Y0gYsq7tOnM.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav
time ./main -l czech -t 12  -m ~/src/whisper.cpp/models/ggml-base.bin -f /home/martin/output.wav -otxt -nt -of cz.txt   -osrt cz_srt -ovtt cz_vtt -olrc cz_lrc 2>&1 | tee cz_tee.txt
