revise the beginning of the README.md for my github project

# Intro

after playing around with fasthtml i got interested in sqlite, sqlite-utils and datasette. i watched some videos and found that fulltext search is easy to implement. so naturally i wanted to try to create a tool to index all pdfs on my system. the full text search works surprisingly well. the found items are sorted somehow by importance. i'm not sure how that works but seems quite good. 

next steps might be to use bag-of-words statistics (or maybe some ai stuff) to separate the documents into different classes and display a 2d map of the clusters.
another thing that would be useful is to read through a document and show corresponding documents that match best to the 500 words surrounding the current position of the reader.


# Prompt

create a python program use pandas, sqlite-utils, pathlib, argparse

1. given one or several input path(s) use pathlib to find recursively all *.pdf files below it(them)
2. create a sqlite database in pdfs.db
3. for each pdf file store the path and the result of `pdftotext -raw -enc UTF-8 <file> -` inside the database. also store the date of when the database entry was created, size in bytes of the pdf file, time of execution for pdftotext, size in bytes of the pdftotext result. list additional information that might come in handy if placed in the database

store also the output of pdfinfo in the dataabse. it looks like this:


Example output
`pdfinfo`

Title:           Teseo-LIV4F GNSS Module - Software manual - User manual
Subject:         The Teseo-LIV4F module family is an easy to use Global Navigation Satellite System (GNSS) stand-alone modules, embedding Teseo single die stand-alone positioning receiver IC working on multiple constellations (GPS/Galileo/Glonass/BeiDou/QZSS).
Author:          STMICROELECTRONICS
Creator:         C2 v20.4.0000 build 240 - c2_rendition_config : Techlit_Active
Producer:        Antenna House PDF Output Library 7.2.1732; modified using iText 2.1.7 by 1T3XT
CreationDate:    Wed Dec 20 10:41:35 2023 CET
ModDate:         Thu Dec 21 13:35:25 2023 CET
Custom Metadata: yes
Metadata Stream: yes
Syntax Error: Suspects object is wrong type (boolean)
Syntax Error: UserProperties object is wrong type (boolean)
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           76
Encrypted:       no
Page size:       595.276 x 841.89 pts (A4)
Page rot:        0
File size:       3663529 bytes
Optimized:       no
PDF version:     1.3

Parse this output and store appropriately in the database. Ignore errors


`pdfinfo -url`
Page  Type          URL
   1  Annotation    mailto:wolf@nereid.pl
   1  Annotation    https://github.com/wolfpld/tracy
  11  Annotation    https://github.com/wolfpld/etcpak
  12  Annotation    https://github.com/wolfpld/tracy

Again, parse this output and store it appropriatly in the database (maybe as another table)

Note: executions of the pdftotext and pdfinfo commands shall run in parallel so that all cores of the computer are busy

# Interrogate Database

## Compare size of PDF with text stored in it

```
sqlite-utils -t pdfs.db "SELECT path, pdf_size, text_size, CAST(pdf_size AS REAL) / text_size AS ratio FROM pdfs ORDER BY ratio DESC;"
```
# References

Datasette： a big bag of tricks for solving interesting problems using SQLite [B55hcKYye_c]
