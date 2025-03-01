from s01_validate_youtube_url import *
assert((("0123456789a")==(validate_youtube_url("https://www.youtube.com/live/0123456789a"))))
assert((("0123456789a")==(validate_youtube_url("https://www.youtube.com/live/0123456789a&abc=123"))))
assert((("_123456789a")==(validate_youtube_url("https://www.youtube.com/watch?v=_123456789a&abc=123"))))
assert((("_123456789a")==(validate_youtube_url("https://youtube.com/watch?v=_123456789a&abc=123"))))
# FIXME: I'm not sure if a Youtube-ID that starts with a - character is handled correctly in the downstream pipeline
assert((("-123456789a")==(validate_youtube_url("https://www.youtu.be/-123456789a&abc=123"))))
assert((("-123456789a")==(validate_youtube_url("https://youtu.be/-123456789a&abc=123"))))
assert(((False)==(validate_youtube_url("http://www.youtube.com/live/0123456789a"))))