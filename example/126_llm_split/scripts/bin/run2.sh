
# ffmpeg -i Codellama\ Deep\ Dive-Y0gYsq7tOnM.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav
time ./main -t 12  -m ~/src/whisper.cpp/models/ggml-base.en.bin -f /home/martin/output.wav -otxt -nt -of distance.txt -osrt distance  | tee distance_tee.txt
