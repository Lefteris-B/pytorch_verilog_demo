import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

class Model:
    def __init__(self, model):
        self.model = model
        self.layers = []

    def __str__(self):
        return '\n'.join(str(layer) for layer in self.layers)

    def parse_layers(self):
        for i, layer in enumerate(self.model.children()):
            if isinstance(layer, nn.Linear):
                self.layers.append(self._convert_linear_layer(layer, i))

    def emit(self):
        verilog_code = []
        for layer in self.layers:
            verilog_code.append(layer.emit())
        return '\n'.join(verilog_code)

    def emit_test_bench(self):
        return test_bench_template

    def _convert_linear_layer(self, layer, index):
        in_features = layer.in_features
        out_features = layer.out_features
        
        # Ensure detachment of tensors
        weight = layer.weight.detach().numpy().T
        bias = layer.bias.detach().numpy()

        return LinearLayer(in_features, out_features, weight, bias, index)

class LinearLayer:
    def __init__(self, in_features, out_features, weight, bias, index):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias
        self.index = index

    def emit(self):
        # Generate Verilog code for the linear layer
        weight_str = self._format_matrix(self.weight)
        bias_str = self._format_vector(self.bias)
        return f"""
        // Linear Layer {self.index}
        module linear_layer_{self.index} (
            input logic [{self.in_features-1}:0] in,
            output logic [{self.out_features-1}:0] out
        );
        // Weight matrix
        logic [{self.in_features-1}:0] weights [{self.out_features-1}:0] = {weight_str};
        // Bias vector
        logic [{self.out_features-1}:0] bias = {bias_str};

        always_comb begin
            out = bias;
            for (int i = 0; i < {self.out_features}; i++) begin
                for (int j = 0; j < {self.in_features}; j++) begin
                    out[i] = out[i] + in[j] * weights[i][j];
                end
            end
        end
        endmodule
        """

    def _format_matrix(self, matrix):
        return '{' + ', '.join(['{' + ', '.join(map(str, row)) + '}' for row in matrix]) + '}'

    def _format_vector(self, vector):
        return '{' + ', '.join(map(str, vector)) + '}'

test_bench_template = """
// Test bench for the generated Verilog code
module test_bench();
    // Define test signals and expected outputs here
endmodule
"""

def convert_model_to_verilog(model_path):
    try:
        # Load the model
        model = SimpleNN()
        model.load_state_dict(torch.load(model_path))

        # Instantiate the Model class with the PyTorch model
        transpiler_model = Model(model)
        
        # Parse the model layers
        transpiler_model.parse_layers()
        
        # Generate the Verilog code
        verilog_code = transpiler_model.emit()
        print(verilog_code)
        
        # Save the Verilog code to a file
        with open('generated_model.v', 'w') as f:
            f.write(verilog_code)
        print('Verilog code saved to generated_model.v')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_model_to_verilog('simple_nn.pth')

