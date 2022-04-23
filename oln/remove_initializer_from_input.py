import onnx
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True, help="input model")
	parser.add_argument("--output", required=True, help="output model")
	args = parser.parse_args()
	return args

# Ref: https://github.com/microsoft/onnxruntime/blob/master/tools/python/remove_initializer_from_input.py
def remove_initializer_from_input(input_path, output_path):

	model = onnx.load(input_path)
	if model.ir_version < 4:
		print(
			'Model with ir_version below 4 requires to include initilizer in graph input'
		)
		return

	inputs = model.graph.input
	name_to_input = {}
	for input in inputs:
		name_to_input[input.name] = input

	for initializer in model.graph.initializer:
		if initializer.name in name_to_input:
			inputs.remove(name_to_input[initializer.name])

	onnx.save(model, output_path)


if __name__ == '__main__':
	
	args = get_args()

	remove_initializer_from_input(args.input, args.output)