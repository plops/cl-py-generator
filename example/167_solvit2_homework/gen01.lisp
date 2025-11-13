(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(progn
  (defparameter *source* "example/167_solvit2_homework/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p01_client"
						   *source*))
   `(do0
     (comments "run with:  export ANTHROPIC_API_KEY=`cat ~/anthropic.key`; uv run python -i p01_client.py")
     (do0 (imports-from (__future__ annotations)
			(claudette *)
			(fastcore.xtras dumps)) ;; 15MB
	  (imports (os
		    ;(pd pandas)
		    ))
	  )
     (setf (aref os.environ (string "ANTHROPIC_LOG"))
	   (string "debug"))
     (print models)

     (comments "['claude-opus-4-1-20250805', 'claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-20250514', 'claude-3-opus-20240229', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022']")

     (comments "haiku is cheapest")
     (setf m (string "claude-haiku-4-5"))
     (setf c (Client m :log True))

     #+nil
     (do0
      (setf r
	    (c (string "Hi there, I am jeremy.")))
      (print r)
      (comments "Message(id='msg_01QeZAHVGFTUUiW6DjCZ3umV', content=[TextBlock(citations=None, text='Hi Jeremy! Nice to meet you. How can I help you today?', type='text')], model='claude-haiku-4-5-20251001', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=In: 14; Out: 18; Cache create: 0; Cache read: 0; Total Tokens: 32; Search: 0)"))

     (do0
      (def adder (a b)
	(declare (type int a b))
	(string "Add two numbers")
	(return (+ a b)))
      (def multer (a b)
	(declare (type int a b))
	(string "Multiply two numbers")
	(return (* a b)))
      (setf tools (list adder multer))

      (setf pr (string "I want to test my tools. Can you use &`[adder,multer]` to solve 42427928*548749+547892?"))

      (setf r (c pr :tools tools))
      (print r)

      (setf tub (dot r (aref content 1))
	    func (aref (globals) tub.name)
	    res (func **tub.input)
	    trc (dictionary :type (string "tool_result")
			    :tool_use_id tub.id
			    :content (dumps res)))
      (comments "

The call of mk_msgs goes to the function mk_msgs_anthropic, which creates a list of messages compatible with the Anthropic API. It uses the @delegates decorator to inherit parameters from mk_msgs, allowing flexible argument passing.
It calls mk_msgs with api='anthropic' to generate the initial list of messages.
If cache_last_ckpt_only is True, it applies _remove_cache_ckpts to each message to strip unnecessary cache checkpoints.
If the message list is empty, it returns early.
Otherwise, it modifies the last message by adding cache control via _add_cache_control, using the cache and ttl parameters.
Finally, it returns the processed list of messages.


The general function, `mk_msgs` (called by mk_msgs_anthropic), creates a list of messages compatible with the OpenAI or Anthropic APIs. It takes a list of message contents (or a single string, which it converts to a list), along with optional arguments and keyword arguments.

- If `msgs` is a string, it wraps it in a list.
- It generates a list `mm` by calling `mk_msg` for each item in `msgs`, assigning alternating roles ('user' for even indices, 'assistant' for odd) and passing through the `api`, `*args`, and `**kw`.
- It then flattens `mm` into `res`: if an item in `mm` is a list, it extends `res` with its elements; otherwise, it appends the item.
- Finally, it returns the flattened list `res`.
")
      (setf msgs (mk_msgs (list pr
				r.content
				(list trc))))

      (setf r (c msgs))

      (comments "
Message(id='msg_01QndYW2KPiotRHiHcUxFjXt', content=[TextBlock(citations=None, text='Now I'll add 547892 to that result:\n', type='text')], model='claude-haiku-4-5-20251001', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=In: 168; Out: 21; Cache create: 0; Cache read: 0; Total Tokens: 189; Search: 0)
")
      (comments "append previous message and the current response to form a new prompt")
      (setf msgs2 (mk_msgs (+ (list (for-generator (m msgs)
						   m.content))
			      (list (dot r content)))))
      (setf r2 (c msgs2))
      
      )
     )
   ))


