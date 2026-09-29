"""Check the deployed model/normalization contract without board SDKs or inference."""

import hashlib
import json
import math
import re
from pathlib import Path

import tflite

ROOT = Path(__file__).resolve().parents[1]


def extract_model(header: str) -> bytes:
    body = re.search(r"sound_model\[\].*?=\s*\{(.*?)\};", header, re.S)
    declared = re.search(r"sound_model_len\s*=\s*(\d+)", header)
    if not body or not declared:
        raise ValueError("Model header declaration is missing")
    data = bytes(int(item, 16) for item in re.findall(r"0x([0-9a-fA-F]{2})\b", body[1]))
    if len(data) != int(declared[1]) or data[4:8] != b"TFL3":
        raise ValueError("Model length or FlatBuffer identifier is invalid")
    return data


def validate_normalization(header: str, count: int) -> None:
    for name in ("mfcc_mean", "mfcc_std"):
        match = re.search(rf"{name}\[(\d+)\]\s*=\s*\{{(.*?)\}};", header, re.S)
        if not match:
            raise ValueError(f"Missing {name} array")
        values = [
            float(v.strip().rstrip("f")) for v in match[2].split(",") if v.strip()
        ]
        if int(match[1]) != count or len(values) != count:
            raise ValueError(f"{name} feature count mismatch")
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name} contains non-finite values")
        if name == "mfcc_std" and not all(v > 0 for v in values):
            raise ValueError("Normalization deviations must be positive")


def validate_model(data: bytes, norm: str, input_scale: float, input_zero: int) -> dict:
    model = tflite.Model.GetRootAsModel(data, 0)
    if model.Version() != 3 or model.SubgraphsLength() != 1:
        raise ValueError("Expected one TFLite v3 subgraph")
    graph = model.Subgraphs(0)
    if graph.InputsLength() != 1 or graph.OutputsLength() != 1:
        raise ValueError("Firmware requires exactly one input and one output")
    for index in (graph.Inputs(0), graph.Outputs(0)):
        if index < 0 or index >= graph.TensorsLength():
            raise ValueError("Input/output tensor index is invalid")
    inp, out = graph.Tensors(graph.Inputs(0)), graph.Tensors(graph.Outputs(0))
    shape = [int(v) for v in inp.ShapeAsNumpy()]
    output_shape = [int(v) for v in out.ShapeAsNumpy()]
    if shape != [1, 39, 61, 1] or output_shape != [1, 1, 1, 6]:
        raise ValueError("Input/output shape no longer matches firmware")
    if inp.Type() != tflite.TensorType.INT8 or out.Type() != tflite.TensorType.INT8:
        raise ValueError("Firmware requires INT8 tensors")
    iq, oq = inp.Quantization(), out.Quantization()
    for name, quant in (("Input", iq), ("Output", oq)):
        if quant is None or quant.ScaleLength() != 1 or quant.ZeroPointLength() != 1:
            raise ValueError(f"{name} requires scalar quantization parameters")
        if not math.isfinite(quant.Scale(0)) or quant.Scale(0) <= 0:
            raise ValueError(f"{name} quantization scale must be finite and positive")
        if not -128 <= quant.ZeroPoint(0) <= 127:
            raise ValueError(f"{name} zero point is outside the INT8 range")
    if (
        not math.isclose(iq.Scale(0), input_scale, rel_tol=1e-6, abs_tol=1e-9)
        or iq.ZeroPoint(0) != input_zero
    ):
        raise ValueError("Input quantization does not match firmware constants")
    if not math.isclose(oq.Scale(0), 1 / 256, abs_tol=1e-9):
        raise ValueError("Output confidence conversion assumes scale 1/256")
    allowed = {
        getattr(tflite.BuiltinOperator, name)
        for name in (
            "CONV_2D",
            "ADD",
            "MUL",
            "MAX_POOL_2D",
            "AVERAGE_POOL_2D",
            "SOFTMAX",
        )
    }
    ops = {
        model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())
    }
    if not ops <= allowed:
        raise ValueError("Model requires an operator absent from the firmware resolver")
    output_producers = []
    for i in range(graph.OperatorsLength()):
        operator = graph.Operators(i)
        opcode_index = operator.OpcodeIndex()
        if opcode_index < 0 or opcode_index >= model.OperatorCodesLength():
            raise ValueError("Operator code index is invalid")
        if graph.Outputs(0) in [
            operator.Outputs(j) for j in range(operator.OutputsLength())
        ]:
            output_producers.append(model.OperatorCodes(opcode_index).BuiltinCode())
    if output_producers != [tflite.BuiltinOperator.SOFTMAX]:
        raise ValueError("Confidence conversion requires a SOFTMAX output producer")
    validate_normalization(norm, 39 * 61)
    return {
        "model_bytes": len(data),
        "model_kib": round(len(data) / 1024, 2),
        "sha256": hashlib.sha256(data).hexdigest(),
        "input_shape": shape,
        "output_shape": output_shape,
        "input_scale": iq.Scale(0),
        "input_zero_point": iq.ZeroPoint(0),
        "output_scale": oq.Scale(0),
        "output_zero_point": oq.ZeroPoint(0),
        "operator_count": len(ops),
        "normalization_features": 2379,
        "scope": "Artifact contract only; no firmware build, inference or accuracy measurement",
    }


def check(root: Path = ROOT) -> dict:
    source = root / "firmware" / "src"
    firmware = (source / "hal_entry.c").read_text()
    scale = re.search(r"#define MODEL_INPUT_SCALE\s+([\d.]+)f", firmware)
    zero = re.search(r"#define MODEL_INPUT_ZERO_POINT\s+\((\d+)\)", firmware)
    if not scale or not zero:
        raise ValueError("Firmware quantization constants are missing")
    return validate_model(
        extract_model((source / "sound_model.h").read_text()),
        (source / "mfcc_norm_params.h").read_text(),
        float(scale[1]),
        int(zero[1]),
    )


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
