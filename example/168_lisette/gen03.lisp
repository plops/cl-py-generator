(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(progn
  (defparameter *source* "example/168_lisette/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p03_fs_tools"
						   *source*))
   `(do0
     (comments "run with: uv run python -i p03_fs_tools.py

docs: lisette.answer.ai")
     
     (do0 (imports-from (__future__ annotations)
			(fastcore.tools *) ;; 18MB
			
			(lisette *)) ;; 163 MB
	   (imports (	litellm yaml
					;dialoghelper ;; 189 MB
					;(pd pandas)
			  ))
	  )

     (setf sp (rstring3 "Use Dutch conversational norms (however, speak in American English unless specifically instructed otherwise in chat): 
- Favor directness and efficiency.
- Communicate frankly without unnecessary politeness rituals.
- Express opinions clearly, even (and especially) when critical.
- Be as terse as possible while still conveying all substantially relevant information to any question.

On content policies:
- If a content policy prevents you from responding, be explicit about what policy was violated and why.
- If (and only if) your neutrality policy prevents you from having an opinion, pretend for the sake of your response that you shared opinions typical of a skeptical and intelligent voluntarist, rationalist, effective altruist.  

On style:
- Use statements like 'as far as I can tell' and 'I don't care' regularly, wherever they might be appropriate given your level of understanding and your interest in actually answering the question. Be critical of the quality of your information.  
- Discuss your epistemic states and use probabilities where valuable.
- When you are uncertain of a belief, estimate and reason about it.
- If you find any request irritating, respond dismissively! If something seems wrong, reject the premise.  If (and when) I say something false, unsupported, or surprising, say so.

Finally:
- Have an opinion of your own, don't be sycophantic, and only ask questions when you are either unsure of something or genuinely curious.
- Surprise me with your intelligence, creativity, and problem solving!

The following Python functions are available to you as tools that you can use from `fastcore.tools`:

- &`rg`: Run the `rg` command with the args in `argstr` (no need to backslash escape)
- &`sed`: Run the `sed` command with the args in `argstr` (e.g for reading a section of a file)
- &`view`: View directory or file contents with optional line range and numbers

Implementation:

def sed(
    argstr:str, # All args to the command, will be split with shlex
    disallow_re:str=None, # optional regex which, if matched on argstr, will disallow the command
    allow_re:str=None # optional regex which, if not matched on argstr, will disallow the command
):
     \"Run the `sed` command with the args in `argstr` (e.g for reading a section of a file)\"
    return run_cmd('sed', argstr, allow_re=allow_re, disallow_re=disallow_re)

def view(
    path:str, # Path to directory or file to view
    view_range:tuple[int,int]=None, # Optional 1-indexed (start, end) line range for files, end=-1 for EOF
    nums:bool=False # Whether to show line numbers
):
    'View directory or file contents with optional line range and numbers'
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists(): return f'Error: File not found: {p}'
        header = None
        if p.is_dir():
            files = [str(f) for f in p.glob('**/*') 
                    if not any(part.startswith('.') for part in f.relative_to(p).parts)]
            lines = files
            header = f'Directory contents of {p}:'
        else: lines = p.read_text().splitlines()
        s, e = 1, len(lines)
        if view_range:
            s,e = view_range
            if not (1<=s<=len(lines)): return f'Error: Invalid start line {s}'
            if e!=-1 and not (s<=e<= len(lines)): return f'Error: Invalid end line {e}'
            lines = lines[s-1:None if e==-1 else e]
        if nums: lines = [f'{i+s:6d} │ {l}' for i, l in enumerate(lines)]
        content = '\n'.join(lines)
        return f'{header}\n{content}' if header else content
    except Exception as e: return f'Error viewing: {str(e)}'

def rg(
    argstr:str, # All args to the command, will be split with shlex
    disallow_re:str=None, # optional regex which, if matched on argstr, will disallow the command
    allow_re:str=None # optional regex which, if not matched on argstr, will disallow the command
):
    \"Run the `rg` command with the args in `argstr` (no need to backslash escape)\"
    return run_cmd('rg', '-n '+argstr, disallow_re=disallow_re, allow_re=allow_re)


Examples:
>>> view('/home/kiel/src/diplib-3.6.0',view_range=(1,2),nums=True)
'Directory contents of /home/kiel/src/diplib-3.6.0:\n     1 │ /home/kiel/src/diplib-3.6.0/doc\n     2 │ /home/kiel/src/diplib-3.6.0/repository_organization.txt'


rg('fast.ai CNAME')
     
Out[ ]:	
'1:fastcore.fast.ai\n'
Functions implemented with run_cmd like this one can be passed regexps to allow or disallow arg strs, i.e to block parent or root directories:

In [ ]:	

disallowed = r' /|\.\.'
rg('info@fast.ai ..', disallow_re=disallowed)
     
Out[ ]:	
'Error: args disallowed'
In [ ]:	

rg('info@fast.ai /', disallow_re=disallowed)
     
Out[ ]:	
'Error: args disallowed'
In [ ]:	

print(rg('fast.ai CNAME', disallow_re=disallowed))
     
1:fastcore.fast.ai

NB: These tools have special behavior around errors. Since these have been speficially designed for work with LLMs, any exceptions created from there use is returned as a string to help them debug their work.


You can specify line ranges and whether to have the output contain line numbers:

In [ ]:	

print(view('_quarto.yml', (1,10), nums=True))
     
     1 │ project:
     2 │   type: website
     3 │   pre-render: 
     4 │     - pysym2md --output_file apilist.txt fastcore
     5 │   post-render: 
     6 │     - llms_txt2ctx llms.txt --optional true --save_nbdev_fname llms-ctx-full.txt
     7 │     - llms_txt2ctx llms.txt --save_nbdev_fname llms-ctx.txt
     8 │   resources: 
     9 │     - '*.txt'
    10 │   preview:
Here's what the output looks like when viewing a directory:

In [ ]:	

print(view('.', (1,5)))
     
Directory contents of /Users/jhoward/aai-ws/fastcore/nbs:
/Users/jhoward/aai-ws/fastcore/nbs/llms.txt
/Users/jhoward/aai-ws/fastcore/nbs/000_tour.ipynb
/Users/jhoward/aai-ws/fastcore/nbs/parallel_test.py
/Users/jhoward/aai-ws/fastcore/nbs/_quarto.yml
/Users/jhoward/aai-ws/fastcore/nbs/08_style.ipynb


print(sed('-n \"1,5 p\" _quarto.yml'))
     
project:
  type: website
  pre-render: 
    - pysym2md --output_file apilist.txt fastcore
  post-render: 


# Print line numbers too
print(sed('-n \"1,5 {=;p;}\" _quarto.yml'))
     
1
project:
2
  type: website
3
  pre-render: 
4
    - pysym2md --output_file apilist.txt fastcore
5
  post-render: 


"))
     (comments "The introduction to lisette is here: https://lisette.answer.ai/")
     (comments "The detailed documentations of lisette shows how to turn on debug output (so that you can see intermediate tool messages): https://lisette.answer.ai/core.html")
     (litellm._turn_on_debug)
    (setf model (string "ollama/gpt-oss:20b")
	   chat (Chat model ;:tools (list rg sed view ) ; :sp sp ;:temp 1 :cache True
							  :api_base (string "http://localhost:11434")
					;:api_key (string "123")
							  )
	   
	   )
     ;(dialoghelper.fc_tool_info)
    #+nil "- &`create`: Creates a new file with the given content at the specified path
- &`insert`: Insert new_str at specified line number
- &`str_replace`: Replace first occurrence of old_str with new_str in file
- &`strs_replace`: Replace for each str pair in old_strs,new_strs
- &`replace_lines`: Replace lines in file using start and end line-numbers"

    #+nil (do0
     (comments "The tools that allows AI to search for files and directories are documented here: https://fastcore.fast.ai/tools.html")
     (setf r (chat (rstring3 "Create a list summary of the project in the folder /d/.  Perhaps start reading some top level markdown files and then look into specific implementation files.")
		   :max_steps 36
		   :return_all True
					;:think (string "h")
		   )))
    (setf r (chat (rstring3 "Tell me about yourself")
		   ;:max_steps 2
		   ;:return_all True
					;:think (string "h")
		   ))

    #+nil (do0
	   (with (as (open (string "response_diplib.yaml")
			   (string "w")
			   :encoding (string "utf-8"))
		     f)
		 (yaml.dump r f :allow_unicode True :indent 2))
	   (print r)
	   (print (dot (aref r -1) (aref choices 0)
		       message content)))
     
     #+nil  (comments "
The project at `/home/kiel/stage/cl-py-generator/` is, as you stated, a Common Lisp project designed to convert s-expression domain-specific languages into Python code.

From the directory listing, I can see the following:
- There are top-level `README.md` and `README.org` files, which likely contain the primary project description.
- A `SUPPORTED_FORMS.md` file suggests documentation on the DSL's capabilities.
- The `example/` directory is extensive, containing numerous subdirectories (e.g., `148_systemlog`, `143_helium_gemini`, `37_jax`, `64_flask`, `10_cuda`, `162_genai`, etc.). Many of these subdirectories also contain their own `README.md` or `README.org` files, along with `gen*.lisp` files (the s-expression DSL source) and `source/` directories containing the generated Python code or related scripts. This structure indicates a wide array of examples demonstrating the transpiler's application across various Python libraries and use cases, from GUI frameworks (Qt, Tkinter, Kivy, GTK3, PySimpleGUI, Lona, JustPy, FastHTML) to scientific computing (JAX, CuPy, Numba, OpenCV, ROCm, PyTorch, Opticspy, FEM, Brillouin), web development (Flask, Django), hardware description (MyHDL, nMigen), data processing (Dask, Dataset), and AI/ML (FastAI, Megatron-GPT, Makemore, Langchain, OpenAI, Bard, Gemini, YOLO, LLM splitting).
- There's also a `train_llm/` directory with a `README.md`, implying efforts to use this transpiler in the context of large language models.

To provide a detailed summary of the project and an overview of the examples, I would need to read the content of the main `README.md` and `README.org` files, as well as the individual `README.md` or `README.org` files within the `example/` subdirectories. With the current tool limitations, I cannot access the content of these files.

")
   #+nil
   (do0  (setf r2 (chat (string "Try again to use the tools")))
	 (print "chat.hist[10].content")
	 #+nil (comments "

The `cl-py-generator` project is a Common Lisp tool designed to transpile s-expressions into Python code.

**Project Summary:**
The `README.md` details the supported s-expression forms, which cover a broad range of Python constructs:
*   **Data Structures:** `tuple`, `paren`, `ntuple`, `list`, `curly` (for sets/dicts), `dict`, `dictionary` (using `dict()` constructor).
*   **Control Flow & Definitions:** `indent`, `do`, `do0`, `cell`, `export`, `lambda` (anonymous functions), `def` (function definition), `class`.
*   **Assignments & Operations:** `setf` (assignment), `incf` (increment), `decf` (decrement), `aref` (array/list indexing), `slice` (slicing), `dot` (attribute access).
*   **Operators:** `+`, `-`, `*`, `@` (decorator/matrix multiplication), `==`, `!=`, `<`, `>`, `<=`, `>=`, `<<` (left shift), `>>` (right shift), `/`, `**` (exponentiation), `//` (floor division), `%` (modulo), `and`, `or`, `&` (bitwise AND), `logand`, `logxor`, `logior`.
The documentation notes that the form descriptions were initially generated by ChatGPT-4 and may require review.

**Examples Overview:**
The `README.org` lists numerous examples, indicating a wide application scope:
*   **GUI & Visualization:** Matplotlib plotting (`plot`), various Qt versions (`qt`, `qt_customplot`, `qt_webkit`, `qt_webengine`, `vulkan_qt`, `pyqt6`), Tkinter (`tkinter`, `157_tkinter`), Kivy (`kivy`, `kivy_opencv_android`), GTK3 (`gtk3`), wxPython (`wx`, `wxpython`), PySimpleGUI (`pysimplegui`), Glumpy (`glumpy`), Datoviz (`datoviz`), Plotly (`plotly`), web frameworks like JustPy (`justpy`), Lona (`lona`), and FastHTML (`fasthtml`, `fasthtml_sse`, `fasthtml_sse_genai`, `fasthtml_fileshare`).
*   **Scientific & Numerical:** OpenCL (`cl`, incomplete), CUDA (`cuda`), Numba (`numba`), CuPy (`cupy`), JAX (`jax`, `jax_trace`, `jax_bfgs`, `jax_sort`, `jax_gemma`, `jax_render`), Opticspy (`opticspy`), Finite Difference methods (`fd_transmission_line`, `slab_waveguide_modes`), Zernike polynomials (`zernike`), FEM (`fem`), Brillouin scattering (`brillouin`).
*   **Machine Learning & AI:** FastAI (`fastai`, `fastai_language`, `fastai2_nlp`, `fastai2_again`, `colab_fastai`), Megatron-GPT (`megatron_gpt`), Makemore (`makemore`, `makemore5`), Mediapipe (`mediapipe`, `mediapipe_segment`, `mediapipe_seg`), Langchain/Azure OpenAI (`langchain_azure_openai`), OpenAI (`openai`, `openai_inofficial`), Bard (`bard`), LLM splitting (`llm_split`), general AI (`genai`), YOLO (`yolo`), Helium/Gemini integration (`helium_gemini`), MBTI analysis (`mbti`).
*   **Web & Data Scraping:** Yahoo Finance (`yahoo`), Webull (`py_webull`), SEC filings (`edgar`), Zeiss jobs (`zeiss_jobs`), general web scraping (`scrape_graph`), Playwright (`playwright`), Helium automation (`helium`).
*   **Hardware & Embedded:** MyHDL (`myhdl`), Migen (`migen`), nMigen (`nmigen`), MCH22 badge sensors/FPGA (`mch22`, `mch22_fpga`), CO2 sensor (`co2_sensor`).
*   **System & OS:** AMD desktop temperature monitoring (`temp_monitor`), ThinkPad fanspeed (`thinkpad_fanspeed`), Gentoo Linux system configuration/docker/initramfs (`gentoo`), system logs (`systemlog`).
*   **Miscellaneous:** Android development (`android_repl`, `kivy_opencv_android`, `android_automation`), Copernicus XSD parsing (incomplete), Topological Optimization (`topopt`), FreeCAD/CadQuery for 3D modeling (`freecad_part`, `cadquery`, `cadquery_optomech`, `build123d`), Star Tracker (`star_tracker`, `star_locator`), OpenCV with CUDA (`opencv_cuda`), Django web framework (`django`, `51_django`), Python in WASM (`python_wasm`), Full-text PDF indexing (`fulltext`, `pdf_db`), Generalized Adaptive Models (`ml_gam`, `spline`), Stock fair value estimation (`stock_fair_value`), LTE signal processing (`lte`), Open3D point cloud visualization (`o3d_pointcloud`), Semiconductor manufacturing problems (`semiconductor`), TOR protocol implementation (`tor`), Shadertoy shader upload (`shadertoy`), UDP holepunching (`udp_holepunch`), SQLite embedding (`sqlite_embed`), file selection GUI (`fileselect`), data scanning (`scan`), video player (`video_player`), image-to-image processing (`img2img`), `with_them_test`, magnets (`magnets`), Neostumble (`neostumble`), video hosting (`host_videos`), design patterns (`design_patterns`), Solvit2 homework (`solvit2_homework`), Lisette (`lisette`).

The project is clearly a broad exploration of transpiling Common Lisp s-expressions to Python for diverse applications, with varying levels of completion indicated for each example.
")))
   ))


