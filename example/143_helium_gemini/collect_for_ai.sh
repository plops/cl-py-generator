for i in gen04.lisp source04/chrome_extension/{manifest.json,popup.{html,js}} source04/tsum/{nginx.conf,pyproject.toml,*.py}; do echo "// start of "$i;cat $i;done|xclip
