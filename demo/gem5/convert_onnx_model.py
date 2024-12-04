#!/bin/env python3
# Script derived and adapted from this source:
# https://iree.dev/guides/ml-frameworks/onnx/#troubleshooting

import argparse
import onnx

parser = argparse.ArgumentParser("ONNX Version Converter")
parser.add_argument("input", type=str, help="Input ONNX file")
parser.add_argument("output", type=str, help="Output ONNX file")
args = parser.parse_args()

original_model = onnx.load_model(args.input)
converted_model = onnx.version_converter.convert_version(original_model, 17)
onnx.save(converted_model, args.output)
