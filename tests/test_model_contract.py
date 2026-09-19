import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_model_contract import ROOT, check, extract_model, validate_normalization, validate_model


class ModelContractTests(unittest.TestCase):
    def test_real_artifacts_match_runtime_contract(self):
        self.assertEqual(check()['model_bytes'], 117136)

    def test_truncated_model_is_rejected(self):
        header = (ROOT/'firmware/src/sound_model.h').read_text()
        with self.assertRaises(ValueError):
            extract_model(header.replace('sound_model_len = 117136', 'sound_model_len = 117135'))

    def test_previous_firmware_scale_is_rejected(self):
        source = ROOT/'firmware/src'
        with self.assertRaisesRegex(ValueError, 'quantization'):
            validate_model(extract_model((source/'sound_model.h').read_text()), (source/'mfcc_norm_params.h').read_text(), 0.088802, 3)

    def test_missing_model_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_model('')

    def test_wrong_feature_count_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_normalization('static const float mfcc_mean[2] = {1.0f};', 2)

    def test_zero_standard_deviation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_normalization('mfcc_mean[1] = {1.0f}; mfcc_std[1] = {0.0f};', 1)

    def test_nonfinite_normalization_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_normalization('mfcc_mean[1] = {nan}; mfcc_std[1] = {1.0f};', 1)


if __name__ == '__main__':
    unittest.main()
