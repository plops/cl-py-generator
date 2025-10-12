					;(ql:quickload "optima")
					;(ql:quickload "alexandria")
(defpackage :cl-py-generator
  (:use :cl
					;:optima
	:alexandria)
  (:export
   ;; public API  
   :emit-py  
   :write-source  
   :write-notebook  
   ;:parse-defun  
   ;:consume-declare  
   ;:print-sufficient-digits-f64  
  
   ;; public state (if you need them externally)  
   ;:*warn-breaking*  
   ;:*file-hashes*  
   ;:*env-functions*  
   ;:*env-macros*  
  
   ;; DSL node names handled by emit-py  
   :tuple :paren :ntuple :list :curly  
   :dict :dictionary  
  
   :indent :do :do0 :class :cell :export :space  
  
   :lambda :def  
  
   := :+ :- :* :@ :== :<< :!= :< :> :<= :>= :>> :/ :** :// :%  
  
   :& :^ :logand :logxor :logior  
   :and :or  
  
   :setf :incf :decf :aref :slice :dot  
  
   :in :is :as
   :not-in :is-not
  
   :comment :comments :symbol  
   :string :string-b :fstring :fstring3 :string3 :rstring3  
  
   :return_ :return  
  
   :for :for-generator :while :if :cond :? :when :unless  
  
   :import :import-from :imports :imports-from  
  
   :with :try :else :finally 
  
   ;; unusual/punctuation tokens used in the code (kept for completeness)  
   ;; :| ;;  i think this is or. not sure if i ever use itisth

))
