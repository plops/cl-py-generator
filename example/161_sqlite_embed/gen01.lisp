(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_data.py"
						 "example/161_sqlite_embed/"))
 `(do0
    (imports ((np numpy)
	      (pd pandas)))
    (imports-from (sqlite_minutils *))
    (setf db (Database (string "/home/kiel/summaries_20250929.db"))
	  tab (db.table (string "items")))))


;; convert a subset of the columns to pandas
;; filter out bad summaries
;; compute 4d and 2d umap
;; compute hdbscan clusters
;; find names for the clusters

;; store the umap so that new emedding entries can be assigned a 2d position and a cluster name

;; visualize the full map

;; display a segment of the full map in the neighborhood of a new embedding entry


