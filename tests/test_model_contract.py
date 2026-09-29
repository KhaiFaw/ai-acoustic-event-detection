import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tflite

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_model_contract import (
    ROOT,
    check,
    extract_model,
    validate_normalization,
    validate_model,
)


class ModelContractTests(unittest.TestCase):
    def model_fixture(self):
        source = ROOT / "firmware/src"
        data = extract_model((source / "sound_model.h").read_text())
        norm = (source / "mfcc_norm_params.h").read_text()
        model = Mock(wraps=tflite.Model.GetRootAsModel(data, 0))
        graph = Mock(wraps=model.Subgraphs(0))
        model.Subgraphs.return_value = graph
        return data, norm, model, graph

    def assert_model_rejected(self, data, norm, model, message):
        with patch(
            "check_model_contract.tflite.Model.GetRootAsModel", return_value=model
        ):
            with self.assertRaisesRegex(ValueError, message):
                validate_model(data, norm, 0.053331278, 3)

    def test_real_artifacts_match_runtime_contract(self):
        self.assertEqual(check()["model_bytes"], 117136)

    def test_truncated_model_is_rejected(self):
        header = (ROOT / "firmware/src/sound_model.h").read_text()
        with self.assertRaises(ValueError):
            extract_model(
                header.replace("sound_model_len = 117136", "sound_model_len = 117135")
            )

    def test_previous_firmware_scale_is_rejected(self):
        source = ROOT / "firmware/src"
        with self.assertRaisesRegex(ValueError, "quantization"):
            validate_model(
                extract_model((source / "sound_model.h").read_text()),
                (source / "mfcc_norm_params.h").read_text(),
                0.088802,
                3,
            )

    def test_missing_model_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_model("")

    def test_wrong_feature_count_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_normalization("static const float mfcc_mean[2] = {1.0f};", 2)

    def test_zero_standard_deviation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_normalization("mfcc_mean[1] = {1.0f}; mfcc_std[1] = {0.0f};", 1)

    def test_nonfinite_normalization_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_normalization("mfcc_mean[1] = {nan}; mfcc_std[1] = {1.0f};", 1)

    def test_extra_input_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        graph.InputsLength.return_value = 2
        self.assert_model_rejected(data, norm, model, "exactly one")

    def test_missing_output_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        graph.OutputsLength.return_value = 0
        self.assert_model_rejected(data, norm, model, "exactly one")

    def test_invalid_tensor_index_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        graph.Inputs.return_value = graph.TensorsLength()
        self.assert_model_rejected(data, norm, model, "tensor index")

    def test_missing_input_quantization_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        tensor = Mock(wraps=graph.Tensors(graph.Inputs(0)))
        tensor.Quantization.return_value = None
        original_tensors = graph.Tensors
        graph.Tensors = lambda index: (
            tensor if index == graph.Inputs(0) else original_tensors(index)
        )
        self.assert_model_rejected(data, norm, model, "scalar quantization")

    def test_invalid_scalar_quantization_is_rejected(self):
        for property_name, value, message in (
            ("ScaleLength", 2, "scalar quantization"),
            ("ZeroPointLength", 0, "scalar quantization"),
            ("Scale", 0.0, "finite and positive"),
            ("Scale", float("nan"), "finite and positive"),
            ("Scale", float("inf"), "finite and positive"),
            ("ZeroPoint", 128, "INT8 range"),
        ):
            for endpoint in ("input", "output"):
                with self.subTest(
                    property=property_name, value=value, endpoint=endpoint
                ):
                    data, norm, model, graph = self.model_fixture()
                    index = graph.Inputs(0) if endpoint == "input" else graph.Outputs(0)
                    tensor = Mock(wraps=graph.Tensors(index))
                    quant = Mock(wraps=tensor.Quantization())
                    getattr(quant, property_name).return_value = value
                    tensor.Quantization.return_value = quant
                    original_tensors = graph.Tensors
                    graph.Tensors = lambda i: (
                        tensor if i == index else original_tensors(i)
                    )
                    self.assert_model_rejected(data, norm, model, message)

    def test_invalid_operator_index_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        operator = Mock(wraps=graph.Operators(0))
        operator.OpcodeIndex.return_value = model.OperatorCodesLength()
        original_operators = graph.Operators
        graph.Operators = lambda i: operator if i == 0 else original_operators(i)
        self.assert_model_rejected(data, norm, model, "Operator code index")

    def test_non_softmax_output_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        index = graph.Operators(graph.OperatorsLength() - 1).OpcodeIndex()
        code = Mock(wraps=model.OperatorCodes(index))
        code.BuiltinCode.return_value = tflite.BuiltinOperator.ADD
        original_codes = model.OperatorCodes
        model.OperatorCodes = lambda i: code if i == index else original_codes(i)
        self.assert_model_rejected(data, norm, model, "SOFTMAX")

    def test_wrong_input_type_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        tensor = Mock(wraps=graph.Tensors(graph.Inputs(0)))
        tensor.Type.return_value = tflite.TensorType.FLOAT32
        original_tensors = graph.Tensors
        graph.Tensors = lambda index: (
            tensor if index == graph.Inputs(0) else original_tensors(index)
        )
        self.assert_model_rejected(data, norm, model, "INT8 tensors")

    def test_wrong_output_scale_is_rejected(self):
        data, norm, model, graph = self.model_fixture()
        tensor = Mock(wraps=graph.Tensors(graph.Outputs(0)))
        quant = Mock(wraps=tensor.Quantization())
        quant.Scale.return_value = 0.5
        tensor.Quantization.return_value = quant
        original_tensors = graph.Tensors
        graph.Tensors = lambda index: (
            tensor if index == graph.Outputs(0) else original_tensors(index)
        )
        self.assert_model_rejected(data, norm, model, "scale 1/256")


if __name__ == "__main__":
    unittest.main()
