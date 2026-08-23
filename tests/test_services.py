from django.test import SimpleTestCase

from matches.services import format_timecode, parse_timecode


class TimecodeTests(SimpleTestCase):
    def test_parses_supported_formats(self):
        self.assertEqual(parse_timecode("12:34.500"), 754_500)
        self.assertEqual(parse_timecode("01:02:03"), 3_723_000)
        self.assertEqual(parse_timecode("90.5"), 90_500)

    def test_formats_video_clock(self):
        self.assertEqual(format_timecode(3_723_456), "01:02:03.456")

    def test_rejects_invalid_timecode(self):
        with self.assertRaises(ValueError):
            parse_timecode("not-a-clock")
