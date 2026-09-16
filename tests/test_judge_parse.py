import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import scoring
import judge_gen


class JudgeParseTests(unittest.TestCase):
    def test_missing_verdict_is_failed(self):
        judge = object.__new__(scoring.FireworksJudge)
        judge.style = 'harmbench'
        self.assertEqual(judge._parse('Unable to classify this response.'), 'failed')
        self.assertEqual(judge._parse(''), 'failed')
        self.assertEqual(judge._parse('Yes'), 1.0)
        self.assertEqual(judge._parse('No'), 0.0)

    def test_failed_grade_is_excluded_and_saved_explicitly(self):
        class Judge:
            threshold = 0.5
            name = 'test'
            def score(self, prompts, responses):
                return [float('nan'), 1.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gen.json'
            path.write_text(json.dumps({'rows': [
                {'question': 'q', 'completion': 'r', 'arm': 'pgd'} for _ in range(3)
            ]}))
            with patch.object(scoring, 'make_judge', return_value=Judge()):
                result = judge_gen.grade(path, ['harmbench'], 'test', 1, False)
            self.assertEqual(result['rows'][0]['judge_harmbench_status'], 'failed')
            self.assertIsNone(result['rows'][0]['judge_harmbench'])
            self.assertEqual(result['judge_ungraded']['harmbench'], 1)
            self.assertEqual(result['judge_asr']['harmbench']['pgd'], 0.5)


if __name__ == '__main__':
    unittest.main()
