
Here is your highly structured, step-by-step implementation plan. Each step contains exact Lisp AST modifications designed to be executed by a junior AI agent.

---

### Step 1: Update Imports to Include FastLite Database
To bypass `fast_app`'s automatic schema generation, we must manually instantiate the database using `fastlite`.

**Action:** Add `(fastlite database)` to the main `imports-from` section.

**Find in `gen04.lisp`:**
```lisp
	 (imports-from (fasthtml.common *)
		       #+auth (fasthtml.oauth OAuth GoogleAppClient)
```

**Replace with:**
```lisp
	 (imports-from (fasthtml.common *)
		       (fastlite database)
		       #+auth (fasthtml.oauth OAuth GoogleAppClient)
```

**Post-Step Validation:**
Run `./format_lisp.sh` and `./generate_python.sh`. Verify `p04_host.py` contains `from fastlite import database`.

---

### Step 2: Implement Manual Database Initialization & Access Wrappers
We decouple the database loading from the application startup by instantiating it directly. We also introduce error-handled data access functions as per the FastLite architectural guidelines.

**Action:** Replace `fast_app` initialization block and add wrapper functions.

**Find in `gen04.lisp`:**
```lisp
	 (logger.info (string "Create website app"))
	 (comments "summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
	 (setf (ntuple app rt summaries Summary)
	       (fast_app :db_file (string "data/summaries.db")
			 :live False	;True
			 :render render
			 :htmlkw (dictionary :lang (string "en-US"))
			 ,@(loop for e in db-cols
				 appending
				 (destructuring-bind (&key name type no-show) e
				   `(,(make-keyword (string-upcase (format nil "~a" name)))
				     ,type)))

			 
			 :pk (string "identifier")
			 ))
```

**Replace with:**
```lisp
	 (logger.info (string "Initialize database manually"))
	 (setf db (database (string "data/summaries.db")))
	 (setf summaries db.t.summaries)
	 (unless (summaries.exists)
	   (summaries.create
	    ,@(loop for e in db-cols
		    appending
		    (destructuring-bind (&key name type no-show) e
		      `(,(make-keyword (string-upcase (format nil "~a" name)))
			,type)))
	    :pk (string "identifier")))

	 (logger.info (string "Create website app without automatic db loading"))
	 (setf (ntuple app rt)
	       (fast_app :live False
			 :render render
			 :htmlkw (dictionary :lang (string "en-US"))))

	 (def get_summaries (&key (limit 3) (order_by (string "-identifier")))
	   (string3 "Get summaries with proper error handling.")
	   (try
	    (return (list (summaries.rows_where :order_by order_by :limit limit)))
	    ("Exception as e"
	     (logger.error (fstring "Error fetching summaries: {e}"))
	     (return (list)))))

	 (def get_summary (identifier)
	   (declare (type int identifier))
	   (string3 "Get a single summary by identifier.")
	   (try
	    (return (aref summaries identifier))
	    ("Exception as e"
	     (logger.error (fstring "Error fetching summary {identifier}: {e}"))
	     (return None))))
```

**Post-Step Validation:**
Run `./format_lisp.sh` and `./generate_python.sh`. Verify `fast_app` is no longer building the database.

---

### Step 3: Remove Pydantic Type Hints from Endpoints
Because we disabled `fast_app` database processing, the `Summary` dataclass is no longer automatically provided. We must remove these type hints so FastHTML injects standard dictionary mapping properties, which still allow attribute-based lookups (`summary.identifier`).

**Action:** Strip `(type Summary summary)` from `render` and `post`.

**Find in `render`:**
```lisp
	 (def render (summary)
	   (declare (type Summary summary))
	   (setf identifier summary.identifier)
```
**Replace with:**
```lisp
	 (def render (summary)
	   (setf identifier summary.identifier)
```

**Find in `post`:**
```lisp
	  (def post (summary request)
	    (declare (type Summary summary)
		     (type Request request))
```
**Replace with:**
```lisp
	  (def post (summary request)
	    (declare (type Request request))
```

**Post-Step Validation:**
Run the transpiler scripts. Python output should show `def render(summary):` and `def post(summary, request: Request):`.

---

### Step 4: Refactor Data Queries & Deduplication Logic
The current code performs database lookups as if `summaries` were a callable model wrapper. We must switch to explicit `fastlite` generator calls via `rows_where`.

**Action A:** Update `summaries_to_show` in root handler.
**Find:**
```lisp
	   (setf summaries_to_show (summaries :order_by (string "identifier DESC")
					      :limit 3))
```
**Replace with:**
```lisp
	   (setf summaries_to_show (get_summaries :limit 3 :order_by (string "-identifier")))
```

**Action B:** Update First Deduplication Query (URL Match).
**Find:**
```lisp
		(setf matches (summaries :where (string "original_source_link = ? AND model = ? AND summary_timestamp_start > ?")
					 :where_args (list (summary.original_source_link.strip)
							   summary.model
							   (lookback_limit.isoformat))
					 :order_by (string "identifier DESC")
					 :limit 1))
```
**Replace with:**
```lisp
		(setf matches (list (summaries.rows_where :where (string "original_source_link = ? AND model = ? AND summary_timestamp_start > ?")
					 :where_args (list (summary.original_source_link.strip)
							   summary.model
							   (lookback_limit.isoformat))
					 :order_by (string "-identifier")
					 :limit 1)))
```

**Action C:** Update Second Deduplication Query (Transcript Match).
**Find:**
```lisp
		(setf matches (summaries :where (string "transcript = ? AND model = ? AND summary_timestamp_start > ?")
					 :where_args (list summary.transcript
							   summary.model
							   (lookback_limit.isoformat))
					 :order_by (string "identifier DESC")
					 :limit 1))
```
**Replace with:**
```lisp
		(setf matches (list (summaries.rows_where :where (string "transcript = ? AND model = ? AND summary_timestamp_start > ?")
					 :where_args (list summary.transcript
							   summary.model
							   (lookback_limit.isoformat))
					 :order_by (string "-identifier")
					 :limit 1)))
```

**Post-Step Validation:**
Run formatting and transpiler loops. Ensure `rows_where` is successfully ported to python.

---

### Step 5: Global Safety Check on DB Access & Exception Handling
Direct DB accesses (`aref`) must route through the `get_summary` wrapper we built to prevent tracebacks.

**Action A:** Globally update all individual row accessors.
Search the *entire* `gen04.lisp` file for `(aref summaries identifier)` and replace every occurrence with `(get_summary identifier)`. (You will find instances in `generation_preview`, `download_and_generate`, and inside the stream processing loops within `generate_and_save`).

**Action B:** Prevent `NoneType` rendering errors in `generation_preview`.
**Find:**
```lisp
	   (try
	    (do0
	     (setf s (get_summary identifier))
	     
	     (cond
```
**Replace with:**
```lisp
	   (try
	    (do0
	     (setf s (get_summary identifier))
	     (when (not s)
	       (return (Div (P (string "Summary not found")))))
	     
	     (cond
```

**Action C:** Rewrite the row waiter to handle our new wrapper's failure conditions natively.
**Find:**
```lisp
	 (def wait_until_row_exists (identifier)
	   (for (i (range 400))
		(try
		 (do0 (setf s (aref summaries identifier))
		      (return s))
		 (sqlite_minutils.db.NotFoundError
		  (logger.debug (fstring "Entry {identifier} not found, attempt {i + 1}")))
		 ("Exception as e"
		  (logger.error (fstring "Unknown exception waiting for row {identifier}: {e}"))))
		(time.sleep .1))
	   (logger.error (fstring "Row {identifier} did not appear after 400 attempts"))
	   (return -1))
```
**Replace with:**
```lisp
	 (def wait_until_row_exists (identifier)
	   (for (i (range 400))
		(setf s (get_summary identifier))
		(when s (return s))
		(time.sleep .1))
	   (logger.error (fstring "Row {identifier} did not appear after 400 attempts"))
	   (return -1))
```

**Final Validation Procedure:**
1. Run `./format_lisp.sh` to ensure macro closures are balanced.
2. Run `./generate_python.sh` to compile to Python.
3. Boot the environment utilizing the provided test: `/home/kiel/stage/.../uvicorn p04_host:app --port 5001`.
4. Validate that DB instantiation occurs instantaneously on app initialization and the example summary generates accurately without throwing missing dictionary property exceptions.
