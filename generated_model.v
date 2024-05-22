
        // Linear Layer 0
        module linear_layer_0 (
            input logic [2:0] in,
            output logic [1:0] out
        );
        // Weight matrix
        logic [2:0] weights [1:0] = {{0.07695621, -0.17262933}, {-0.15372275, 0.36823773}, {0.18003024, -0.3102513}};
        // Bias vector
        logic [1:0] bias = {0.1510171, -0.42589796};

        always_comb begin
            out = bias;
            for (int i = 0; i < 2; i++) begin
                for (int j = 0; j < 3; j++) begin
                    out[i] = out[i] + in[j] * weights[i][j];
                end
            end
        end
        endmodule
        