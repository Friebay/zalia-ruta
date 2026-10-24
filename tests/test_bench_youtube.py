import json
from zalia_ruta.bench.youtube import parse_youtube_json3


def test_parse_youtube_json3():
    sample_json = {
        "events": [
            {
                "tStartMs": 500,
                "dDurationMs": 2000,
                "segs": [
                    {"utf8": "Labas"},
                    {"utf8": " vakaras", "tOffsetMs": 300},
                ],
            }
        ]
    }

    words = parse_youtube_json3(sample_json)
    assert len(words) == 2
    assert words[0].word == "Labas"
    assert words[0].start == 0.5
    assert words[1].word == "vakaras"
    assert words[1].start == 0.8
