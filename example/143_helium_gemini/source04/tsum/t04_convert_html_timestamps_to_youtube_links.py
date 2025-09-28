from s04_convert_html_timestamps_to_youtube_links import replace_timestamps_in_html

def test_mm_ss_replacement():
    youtube = "https://www.youtube.com/watch?v=8S4a_LdHhsc"
    html = '<p><strong>14:58 Paper 1:</strong></p>'
    out = replace_timestamps_in_html(html, youtube)
    # 14*60 + 58 = 898
    assert 't=898s' in out
    assert '<a href="' in out and '14:58' in out

def test_hh_mm_ss_replacement():
    youtube = "https://www.youtube.com/watch?v=8S4a_LdHhsc"
    html = '<p><strong>01:03:05 Testing:</strong></p>'
    out = replace_timestamps_in_html(html, youtube)
    # 1*3600 + 3*60 + 5 = 3785
    assert 't=3785s' in out
    assert '<a href="' in out and '01:03:05' in out

def test_multiple_timestamps_and_url_normalization():
    # input URL contains an existing time param which should be ignored after normalization
    youtube = "https://youtu.be/8S4a_LdHhsc?t=100"
    html = """
    <p><strong>00:03:48 Debunking:</strong></p>
    <p><strong>14:58 Paper 1:</strong></p>
    <p><strong>01:06:01 Targeting Apoptosis:</strong></p>
    """
    out = replace_timestamps_in_html(html, youtube)
    # expected seconds:
    # 00:03:48 -> 3*60 + 48 = 228
    # 14:58 -> 14*60 + 58 = 898
    # 01:06:01 -> 1*3600 + 6*60 + 1 = 3961
    assert out.count('<a href="') == 3
    assert 't=228s' in out
    assert 't=898s' in out
    assert 't=3961s' in out
    # ensure the original t=100s from input url is not present
    assert 't=100s' not in out
    # ensure links point to the canonical watch?v=ID form
    assert 'watch?v=8S4a_LdHhsc' in out

def test_invalid_url_no_change():
    bad = "https://example.com/watch?v=xxxx"
    html = "<div><p><strong>01:00 Sample:</strong></p></div>"
    out = replace_timestamps_in_html(html, bad)
    # should be unchanged: no anchor tags and original timestamp text remains
    assert out == html
    assert '<a href="' not in out
    assert '01:00' in out

if __name__ == "__main__":
    # quick manual run
    test_mm_ss_replacement()
    test_hh_mm_ss_replacement()
    test_multiple_timestamps_and_url_normalization()
    test_invalid_url_no_change()
    print("ok")
