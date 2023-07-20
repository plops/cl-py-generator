revise the following text for inclusion in a readme of a github project: `i tried to run the 13B llama 2 model on my laptop (AMD Ryzen 5 5625, 16GB RAM). here are my notes of how to install it:`



```

python -m venv ~/pythonai
. ~/pythonai/bin/activate

python3 -m pip install -r requirements.txt # numpy sentencepiece
python3 convert.py /mnt/llama/llama-2-13b-chat/ 


./quantize /mnt/llama/llama-2-13b-chat/ggml-model-f32.bin /mnt/llama/llama-2-13b-chat/ggml-model-q4_0.bin q4_0

# 7GB file
```
```

./main -ins \
-f ./prompts/alpaca.txt \
-t 8 \
-m /mnt/llama/llama-2-13b-chat/ggml-model-q4_0.bin \
--color \
-c 2048 \
--temp 0.7 \
--repeat_penalty 1.1 \
-s 42 \
-n -1



```

- example run:


``` Below is an instruction that describes a task. Write a response that appropriately completes the request.
> revise the following text for inclusion in a readme of a github project: `i tried to run the 13B llama 2 model on my laptop (AMD Ryzen 5 5625, 16GB RAM). here are my notes of how to install it:`
If you're looking to run the 13B Llama 2 model on your laptop with an AMD Ryzen 5 5625 processor and 16GB of RAM, here are some notes on how to install it:

First, make sure that you have the necessary dependencies installed. You'll need Python 3.7 or later, as well as the following packages:

* `numpy`
* `scipy`

> 

llama_print_timings:        load time =   722.40 ms
llama_print_timings:      sample time =    58.91 ms /    99 runs   (    0.60 ms per token,  1680.53 tokens per second)
llama_print_timings: prompt eval time = 20467.02 ms /   103 tokens (  198.71 ms per token,     5.03 tokens per second)
llama_print_timings:        eval time = 26071.94 ms /    99 runs   (  263.35 ms per token,     3.80 tokens per second)
llama_print_timings:       total time = 56728.33 ms
```

- i find it a bit slow. especially the initial delay after i entered my prompt


some links:
- https://www.youtube.com/watch?v=19MfOJjkyg0
- https://github.com/ggerganov/llama.cpp


- contents of alpaca.txt:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.
```
