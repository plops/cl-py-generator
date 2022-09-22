(ql:quickload "read-csv")

(use-package :read-csv)

(progn
  (defparameter *df* (with-open-file (s "source/links_covid_update.csv")
		       (parse-csv s)))

 (with-open-file (s "source/01_download_audio.sh"
		      :direction :output
		      :if-does-not-exist :create
		      :if-exists :supersede)
   (loop for (idx text location href)
	 in (rest *df*)
	 do
	 (format
	  s
	  "youtube-dl -t -f 251 ~a -o \"~3,'0d_%(title)s-%(id)s.%(ext)s\" ~%"
	  (read-from-string idx)
	  href))
   ))
