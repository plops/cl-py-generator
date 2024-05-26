# install

```
# install tbselenium with dependencies
cd 136_tbs
python -m venv venv
. venv/bin/activate
pip install tbselenium

# install geckodriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.31.0/geckodriver-v0.31.0-linux64.tar.gz
tar xaf geckodriver*.tar.gz
rm geckodriver*.tar.gz
mv geckodriver venv/bin


```

# References

https://github.com/mozilla/geckodriver/
