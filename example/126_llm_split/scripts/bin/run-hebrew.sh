
# ffmpeg -i Codellama\ Deep\ Dive-Y0gYsq7tOnM.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav
time ./main -l hebrew -t 12  -m ~/src/whisper.cpp/models/ggml-base.bin -f /home/martin/output.wav -otxt -nt -of hebrew.txt   -osrt hebrew -ovtt hebrew_vtt -olrc hebrew_lrc 2>&1 | tee hebrew_tee.txt
