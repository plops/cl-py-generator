# install

```
# install tbselenium with dependencies
cd 136_tbs
python -m venv venv
. venv/bin/activate
pip install tbselenium keyboard

# install geckodriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz
tar xaf geckodriver*.tar.gz
rm geckodriver*.tar.gz
mv geckodriver venv/bin # easiest way to get geckodriver into PATH
```

- in order for the the python script to work, start tor browser with
  the flag `-marionette`:
  
```
tor-browser $ ./start-tor-browser.desktop -marionette
```

- the python script can visit sites but it can't download files to diskccy

# References

https://github.com/mozilla/geckodriver/

- thorough library to control firefox that contains download and downloads
https://github.com/david-dick/firefox-marionette/blob/master/lib/Firefox/Marionette.pm

- how to get state of download by looking at about:downloads
https://stackoverflow.com/questions/35891393/how-to-get-file-download-complete-status-using-selenium-web-driver-c-sharp
