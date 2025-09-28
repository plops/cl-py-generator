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

if __name__ == "__main__":
    # quick manual run
    test_mm_ss_replacement()
    test_hh_mm_ss_replacement()
    print("ok")

