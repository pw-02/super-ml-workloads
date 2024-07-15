import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame from the provided data
data = {
    "name": [
        "arange", "boolean_identity", "1l_reshape", "1l_flatten", "tril", "1l_slice", "1l_pad", "triu", "bitshift", 
        "1l_linear", "bitwise_ops", "1l_tiny_div", "blackman_window", "quantize_dequantize", "eye", "1l_identity", 
        "1l_batch_norm", "1l_sqrt", "1l_leakyrelu", "1l_topk", "1l_downsample", "boolean", "max", "1l_upsample", 
        "min", "1l_mean", "1l_concat", "1l_var", "1l_div", "gather_elements", "1l_conv", "1l_average", "1l_where", 
        "rounding_ops", "1l_prelu", "1l_elu", "2l_sigmoid_small", "1l_erf", "1l_sigmoid", "1l_powf", "hard_swish", 
        "reducel1", "2l_relu_fc", "less", "scatter_nd", "hard_max", "clip", "hard_sigmoid", "1l_gelu_noappx", 
        "log_softmax", "1l_gelu_tanh_appx", "selu", "scatter_elements", "softsign", "2l_relu_sigmoid_small", "rnn", 
        "reducel2", "1l_eltwise_div", "2l_relu_sigmoid", "tutorial", "1l_conv_transpose", "1l_max_pool", "1l_lppool", 
        "remainder", "1l_relu", "1l_tanh", "2l_relu_small", "layernorm", "ltsf", "linear_svc", "xgboost_reg", "celu", 
        "1l_mlp", "softplus", "1l_softmax", "2l_relu_sigmoid_conv", "prelu_gmm", "hummingbird_decision_tree", 
        "sklearn_mlp", "mish", "4l_relu_conv_fc", "logsumexp", "3l_relu_conv_fc", "3l_relu_conv_fc", "gru", "lstm", 
        "gather_nd", "trig", "oh_decision_tree", "idolmodel", "lightgbm", "gradient_boosted_trees", "xgboost", 
        "lstm_medium", "mnist_classifier", "decision_tree", "lstm_large", "random_forest", "little_transformer"
    ],
    "total_time": [
        0.028655949, 0.032943079, 0.036804001, 0.038762182, 0.039407958, 0.040275086, 0.041514748, 0.04334649, 
        0.04578185, 0.048664476, 0.053456415, 0.056095883, 0.058223393, 0.063560496, 0.064647623, 0.069041135, 
        0.074520453, 0.079008478, 0.083816845, 0.084396003, 0.087391277, 0.090427397, 0.092109681, 0.10289815, 
        0.106323152, 0.106809642, 0.115937654, 0.119935071, 0.126656536, 0.130793665, 0.130978449, 0.136773763, 
        0.145234659, 0.16214093, 0.163052741, 0.203431273, 0.225013461, 0.229247457, 0.241892342, 0.247605005, 
        0.249985384, 0.266619296, 0.28183462, 0.301218098, 0.325894552, 0.326219402, 0.370784641, 0.391783353, 
        0.39433732, 0.407501618, 0.440746925, 0.453386725, 0.485731247, 0.491160289, 0.510754921, 0.522613958, 
        0.541967533, 0.545804538, 0.574801101, 0.581892637, 0.59155386, 0.610595763, 0.620497257, 0.667819128, 
        0.674975759, 0.697675336, 0.710597452, 0.752636404, 0.753189081, 0.79721038, 0.81009639, 0.83664699, 
        0.944059275, 0.963719949, 1.050375876, 1.065338708, 1.182060809, 1.249266646, 1.273863171, 1.28052759, 
        1.620146721, 1.829670215, 1.888416898, 1.900541202, 1.944665916, 2.208445135, 2.61587715, 3.162645661, 
        4.610802473, 5.525159346, 5.844293695, 9.882124233, 11.02279841, 18.75728508, 21.40613681, 30.636711, 
        51.24668235, 54.65276266, 117.1872825
    ],
    "num_rows": [
        2, 3, 6, 18, 13, 7, 18, 13, 18, 3, 16, 12, 12, 43, 40, 2, 24, 4, 4, 40, 143, 39, 35, 225, 35, 301, 108, 40, 
        1, 208, 10, 130, 448, 81, 66, 99, 2, 4, 4, 6, 12, 56, 8, 10, 94, 24, 64, 40, 30, 40, 55, 57, 1482, 96, 6, 
        38, 133, 17, 384, 526, 2984, 249, 91, 57, 4, 4, 6, 637, 4070, 3078, 278, 28, 24, 20, 35, 6776, 6546, 111, 
        43, 28, 8617, 64, 14352, 14352, 130, 164, 1478, 135, 111, 33510, 1104, 2746, 1129, 92989, 167584, 132, 
        307860, 16736, 500574
    ]
}

df = pd.DataFrame(data)

# Create the scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x="num_rows", y="total_time", hue="total_time", palette="viridis", size="total_time", sizes=(20, 200), legend=None)

# Add titles and labels
plt.title("Total Time vs Number of Rows")
plt.xlabel("Number of Rows")
plt.ylabel("Total Time (s)")

# Show the plot
plt.show()
