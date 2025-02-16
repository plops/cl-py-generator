from s01_validate_youtube_url import *
assert((("0123456789a")==(validate_youtube_url("https://www.youtube.com/live/0123456789a"))))
assert((("0123456789a")==(validate_youtube_url("https://www.youtube.com/live/0123456789a&abc=123"))))
assert(((False)==(validate_youtube_url("http://www.youtube.com/live/0123456789a"))))