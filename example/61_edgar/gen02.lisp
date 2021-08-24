(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)
;; https://pypi.org/project/python-edgar/
;; python3 -m pip install --user  git+https://github.com/edgarminers/python-edgar

;; limit to 10 requests per second

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/61_edgar")
  (defparameter *code-file* "run_02_parse_edgar")
  (defparameter *source* (format nil "~a/source/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
		   (format (- (time.time) start_time)
			   ,@rest)))))
  (let* ((l-concept #+nil `(Assets
		      Liabilities
		      AvailableForSaleSecurities
		      CommonStockSharesAuthorized
		      CommonStockSharesIssued
		      CommonStockSharesOutstanding
		      CommonStockValue
		      ConvertibleDebtEquity
		      DebtCurrent
		      )
		    `(Assets
							     AssetsCurrent
							     AvailableForSaleSecurities
							     CommonStockSharesAuthorized
							     CommonStockSharesIssued
							     CommonStockSharesOutstanding
							     CommonStockValue
							     CustomerAdvancesCurrent
							     DebtCurrent
							     EmployeeRelatedLiabilitiesCurrent
							     Goodwill
							     IncomeTaxesReceivable
							     InventoryNet
							     Liabilities
							     MinorityInterest
							     StockholdersEquity
							     ConvertibleDebtEquity
							     OtherReceivables
							     TreasuryStockShares
							     TreasuryStockValue
							     UnrecognizedTaxBenefits
							     LongTermDebt
							     LongTermDebtCurrent
							     DeferredRevenue
							     ;EntityPublicFloat
							     AssetsFairValueDisclosure
							     ))
	 (code
	   `(do0
	     (do0
		  
		     (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		     (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation) 
					;(xrp xarray.plot)
			       ))
                  
		     (plt.ion)
					;(plt.ioff)
		     (setf font (dict ((string size) (string 5))))
		     (matplotlib.rc (string "font") **font)
		     )
	     (imports (			;os
					;sys
					;time
					;docopt
					pathlib
					;(np numpy)
					;serial
		       (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
		       scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
		       (np numpy)
					; scipy.sparse
					;scipy.sparse.linalg
					; jax
					;jax.random
					;jax.config
					;copy
		       subprocess
		       threading
					;datetime
					;time
		       ; mss
		       ;cv2
		       time
		       ;edgar
		       tqdm
					;requests
		       ;xsdata
					;generated
		     ;  xbrl
		       ))
	     (imports (logging
		       xbrl
		       xbrl.cache
		       xbrl.instance))
	     ;"from generated import *"
	     ;"from xsdata.formats.dataclass.parsers import XmlParser"
	     
	     
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"))
	      _code_generation_time
	      (string ,(multiple-value-bind
			     (second minute hour date month year day-of-week dst-p tz)
			   (get-decoded-time)
			 (declare (ignorable dst-p))
			 (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				 hour
				 minute
				 second
				 (nth day-of-week *day-names*)
				 year
				 month
				 date
				 (- tz)))))
	     (setf start_time (time.time)
		   debug True)
	     (setf fns ("list"
			(dot 
			 (pathlib.Path (string "xml_all/"))
			 (glob (string "mu-20??????.xml")))))
	     
	     (do0
	      (setf user_agent (string "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36"))
	      (logging.basicConfig :level logging.INFO)
	      (do0 
			    (comments "py-xbrl")
			    (setf cache (xbrl.cache.HttpCache (string "./cache")))
			    (cache.set_connection_params
			     :delay 500
			     :retries 5
			     :backoff_factor .8
			     :logs True)
			    (cache.set_headers (dict ((string "From") (string "your.name@company.com"))
						     ((string "User-Agent") user_agent)))
			    (setf parser (xbrl.instance.XbrlParser cache
								   )
				  )))
	    #+nil (setf facts (list))
	     (setf facts_to_collect (list ,@(loop for e in l-concept
						  #+nil
						  `(Assets
						    Liabilities
						    ;ShareholdersEquity
						    )
						  #+nil `(Assets
							     AssetsCurrent
							     AvailableForSaleSecurities
							     CommonStockSharesAuthorized
							     CommonStockSharesIssued
							     CommonStockSharesOutstanding
							     CommonStockValue
							     CustomerAdvancesCurrent
							     DebtCurrent
							     EmployeeRelatedLiabilitiesCurrent
							     Goodwill
							     IncomeTaxesReceivable
							     InventoryNet
							     Liabilities
							     MinorityInterest
							     StockholdersEquity
							     ConvertibleDebtEquity
							     OtherReceivables
							     TreasuryStockShares
							     TreasuryStockValue
							     UnrecognizedTaxBenefits
							     LongTermDebt
							     LongTermDebtCurrent
							     DeferredRevenue
							     EntityPublicFloat
							     AssetsFairValueDisclosure
							     )
						  
					#+nil	  `(Assets


								AssetsCurrent
								AssetsFairValueDisclosure
								AvailableForSaleSecurities
								BasisOfAccounting
								CommonStockSharesAuthorized
								CommonStockSharesIssued
								CommonStockSharesOutstanding
								CommonStockValue
								ComprehensiveIncomeNetOfTax
								ConvertibleDebtEquity
								CostOfGoodsAndServicesSold
								CustomerAdvancesCurrent
								DebtCurrent
								DeferredRevenue
								Depreciation
								DepreciationDepletionAndAmortization
								DerivativeAssets
								DerivativeLiabilities
								EarningsPerShareBasic
								EarningsPerShareDiluted
								EmployeeRelatedLiabilitiesCurrent
								EntityPublicFloat
								EscrowDeposit
								Goodwill
								GrossProfit
								IncomeTaxExpenseBenefit
								IncomeTaxesPaidNet
								IncomeTaxesReceivable
								InterestExpense
								InventoryNet
								InvestmentIncomeNet
								Liabilities
								LongTermDebt
								LongTermDebtCurrent
								MinorityInterest
								OperatingIncomeLoss
								OperatingLossCarryforwards
								OtherReceivables
								PreOpeningCosts
								ProfitLoss
								RevenueFromGrants
								RoyaltyRevenue
								SaleOfStockPricePerShare
								SalesRevenueNet
								SeveranceCosts1
								ShareBasedCompensation
								StockholdersEquity
								TreasuryStockShares
								TreasuryStockValue
								UnrecognizedTaxBenefits)
							    collect
							    `(string ,e))))
	     (do0 (setf res (list))
		  (for (fn ;(list (aref fns 0))
			   (tqdm.tqdm fns)
			   )
		       ,(lprint "read" `(fn))
		       
		       (do0
			
			(try
			   (do0
			    (do0
			     (comments "py-xbrl")
			     (setf inst (parser.parse_instance_locally (str fn)))
			     
			   

			     (for (fact inst.facts)
				  #+nil (facts.append fact.concept.name)
				  (unless (in fact.concept.name
					      facts_to_collect)
				    continue)
				  (when (< 0 (len fact.context.segments))
				    continue)
				  (try
				   (do0
				    (setf d (dictionary :filename (str fn)))
				    ,@(loop for (e f) in `((date fact.context.instant_date)
							   (concept fact.concept.name)
							   (value fact.value)
							   (unit fact.unit.unit_id)
							   (decimals fact.decimals)
							   )
					    collect
					    `(do0
					      (try
					       (do0
						(setf (aref d (string ,e))
						      ,f))
					       ("Exception as e"
						,(lprint (format nil "exception ~a" e)
							 `(e fn))))))
				    
				    (res.append d))
				   ("Exception as e"
				    ,(lprint "exception200" `(e fact.concept.name fn))
				    pass)))

			     
			     )
			    #+nil (do0
				   (comments "python-xbrl")
				   (setf parser (xbrl.XBRLParser)
					 xb (parser.parse (open fn))
					 gaap (parser.parseGAAP xb
								:doc_date (dot fn 
									       stem
									       (aref (split (string "-"))
										     -1))
								:context (string "current")
								:ignore_errors 0)
					 seri (xbrl.GAAPSerializer)
					 d (seri.dump gaap)
					 )
				   (setf (aref d (string "filename"))
					 fn)
				   (res.append d)
				   ))
			   ("Exception as e"
			    ,(lprint "exception" `(e fn))
			  
			    (res.append (dictionary :filename fn
						    :comment e))
			   
			    pass)) 
			
			)))
	  #+nil    ,(let ((l `(Assets


		      AssetsCurrent
		      AssetsFairValueDisclosure
		      AvailableForSaleSecurities
		      BasisOfAccounting
		      CommonStockSharesAuthorized
		      CommonStockSharesIssued
		      CommonStockSharesOutstanding
		      CommonStockValue
		      ComprehensiveIncomeNetOfTax
		      ConvertibleDebtEquity
		      CostOfGoodsAndServicesSold
		      CustomerAdvancesCurrent
		      DebtCurrent
		      DeferredRevenue
		      Depreciation
		      DepreciationDepletionAndAmortization
		      DerivativeAssets
		      DerivativeLiabilities
		      EarningsPerShareBasic
		      EarningsPerShareDiluted
		      EmployeeRelatedLiabilitiesCurrent
		      EntityPublicFloat
		      EscrowDeposit
		      Goodwill
		      GrossProfit
		      IncomeTaxExpenseBenefit
		      IncomeTaxesPaidNet
		      IncomeTaxesReceivable
		      InterestExpense
		      InventoryNet
		      InvestmentIncomeNet
		      Liabilities
		      LongTermDebt
		      LongTermDebtCurrent
		      MinorityInterest
		      OperatingIncomeLoss
		      OperatingLossCarryforwards
		      OtherReceivables
		      PreOpeningCosts
		      ProfitLoss
		      RevenueFromGrants
		      RoyaltyRevenue
		      SaleOfStockPricePerShare
		      SalesRevenueNet
		      SeveranceCosts1
		      ShareBasedCompensation
		      StockholdersEquity
		      TreasuryStockShares
		      TreasuryStockValue
		      UnrecognizedTaxBenefits)
		      ))
		`(do0
		  (def get (n &key fns)
		    (setf facts_to_collect (list ,@(loop for e in l collect `(string ,e))))
		    (setf res (list))
		    (for (fn (tqdm.tqdm fns))
			 (setf inst (parser.parse_instance_locally (str fn)))
		     (for (fact inst.facts)
			  (unless (in fact.concept.name
				      (list (aref facts_to_collect n)))
			    continue)
			  (when (< 0 (len fact.context.segments))
			    continue)
			  (try
			   (res.append (dictionary :filename (str fn)
						   :date fact.context.instant_date
						   :concept fact.concept.name
						   :value fact.value))
			   ("Exception as e"
			    ,(lprint "exception354" `(e fact.concept.name fn))
			    pass))))
		    (return (pd.DataFrame res)))
		 ))

	     
	     (setf df0 (pd.DataFrame res)
		   )

	     (do0
	      (setf pdf (dot (aref df0 (list (string "date")
					     (string "concept")
					     (string "value")
					     (string "unit")
					     (string "decimals")))
			     (aref iloc (dot (aref df0 (list (string "date")
							     (string "concept")))
					     (drop_duplicates)
					     index))
			     (pivot :index (string "date")
				    :columns (string "concept")))))

	     #+nl(do0
	      (setf df1 (dot (aref df0 (list (string "date")
					   (string "concept")
					     (string "value"))) (drop_duplicates))
		    df2 (aref df0.iloc df1.index))
	      (setf pivot_df (df1.pivot :index (string "date")
				       :columns (string "concept")))
	      (print pivot_df))
	     
	     #+nil
	     (do0 (do0
	       (comments "remove columns that only contain zeros")
	       (setf df (dot df (aref loc ":"
				      (dot (!= df 0)
					   (any :axis 0))))))
		  (setf df (df.sort_values :by (string "filename"))))
	     (df0.to_csv (string "mu.csv"))
	     (pdf.to_csv (string "mu_pivot.csv"))

	     #+nil (do0
	      (plt.figure :figsize (list 12 8))
	      (setf pl (list 1 3))
	      ,@(loop for e in `(Assets
				 Liabilities
				 ;ShareholdersEquity
				 )
		      and ei from 0
		      collect
		      `(do0
			(setf ax (plt.subplot2grid pl (list 0 0)))
			(pivot_df.plot :y (tuple (string "value")
						 (string ,e))
				       :ax ax)
			(plt.grid))))

	     ,(let ((n-plots 4))
	      `(do0
	       (plt.figure :figsize (list 14 11))
	       (setf pl (list ,n-plots 1))
	       ,@(loop for i below n-plots collect
		       `(setf ,(format nil "ax~a" i)
			      (plt.subplot2grid pl (list ,i 0))))

	       (def unit_selector (x)
		 (unless (isinstance x str)
		   (return False))
		 (when
		     (x.startswith (string "u0"))
		   (return False))
		 (return True)
		 )
	       ,@(loop for e in l-concept #+nil `(Assets
						  Liabilities)
		       and ei from 0
		       collect
		       `(do0
			 (setf all_units (dot (aref pdf (tuple (string "unit")
							       (string ,e)))
					      (unique)
					      (tolist)
					      
					      ))
			 
			 (setf current_unit
			       (next
				(filter unit_selector
					all_units))
			       )
			 (setf da (aref pdf (tuple (string "value")
						   (string ,e))))
			 (plt.sca ax0)
			 ,@(loop for e in `((usd :mi 3e9) (usd :ma 3e9) (shares :mi 0))
				 and ei from 1
				 collect
				 (destructuring-bind (unit-name &key mi ma) e
				   `(do0
				     (when (and (== current_unit (string ,unit-name))
						,(if mi
						   `(< ,mi (da.max))
						   `True)
						,(if ma
						   `(< (da.max) ,ma)
						   `True))
				       
				       (plt.sca ,(format nil "ax~a" ei))))))
			 (plt.plot (dot pdf index)
				   da
				   :label (dot (string ,(format nil "~a [{}]" e))
					       (format current_unit
						       )))
			 ))
	       ,@(loop for i below n-plots collect
		       `(do0
			 (plt.sca ,(format nil "ax~a" i))
			 (plt.xlabel (string "date"))
			 (plt.legend)
			 (plt.grid)))
	       )))
	   ))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

