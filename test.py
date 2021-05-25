import unittest
import workexp


class TestExp(unittest.TestCase):

    def test_default(self):
        workexp.ex.run(named_configs=['test'])

    def test_accall(self):
        workexp.ex.run(named_configs=['accall', 'test'])

    def test_mall(self):
        workexp.ex.run(named_configs=['mall', 'test'])

    def test_rall(self):
        workexp.ex.run(named_configs=['test', "tuned"], config_updates={
                       'run': [0, 1, 2, 3, 4, 5, 6]})
        # workexp.ex.run(named_configs=['test', 'rall'])


if __name__ == '__main__':
    unittest.main()
