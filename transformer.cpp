/*
    transformer.cpp

    This program implements a simplified transformer block using only C++ and the standard library.
    It creates synthetic data and passes it through a transformer block that consists of:
      - A self-attention layer (using a single attention head)
      - A feed-forward network (two linear layers with a ReLU activation)
      - Residual connections are added after both sub-layers.

    The code includes:
      - A Matrix class that implements basic matrix operations (multiplication, addition, transpose, etc.)
      - A Linear class that represents a fully connected (dense) layer (with weights and biases)
      - SelfAttention, FeedForward, and TransformerBlock classes that combine the above to mimic a transformer block.
      - A main() function that creates synthetic data (a small matrix), processes it through the transformer block, and prints the result.
    
    Note: This example is meant for demonstration/educational purposes and is not optimized for performance or training.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// ------------------------- Matrix Class -------------------------
// This class encapsulates a 2D matrix (using vector of vector of double)
// and provides basic operations like random initialization, printing,
// dot product (matrix multiplication), addition, transpose, ReLU, and softmax.
class Matrix {
public:
    vector<vector<double>> data;
    int rows;
    int cols;
    
    // Constructor to initialize a matrix with given rows and columns.
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        // Initialize the matrix with zeros.
        data.resize(rows, vector<double>(cols, 0.0));
    }
    
    // Fill the matrix with random values between -1 and 1.
    void randomize() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // ((rand / RAND_MAX) * 2 - 1) gives a random double in [-1, 1]
                data[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }
    
    // Print the matrix to standard output.
    void print() {
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                cout << data[i][j] << "\t";
            }
            cout << endl;
        }
    }
    
    // Matrix multiplication (dot product).
    // Multiplies this matrix with another matrix "other".
    Matrix dot(const Matrix &other) const {
        if (cols != other.rows) {
            cout << "Matrix dimensions do not match for dot product." << endl;
            exit(1);
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < other.cols; j++){
                double sum = 0;
                for (int k = 0; k < cols; k++){
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }
    
    // Element-wise matrix addition.
    Matrix add(const Matrix &other) const {
        if (rows != other.rows || cols != other.cols) {
            cout << "Matrix dimensions do not match for addition." << endl;
            exit(1);
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    // Apply the ReLU activation function element-wise.
    Matrix relu() const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                result.data[i][j] = (data[i][j] > 0) ? data[i][j] : 0;
            }
        }
        return result;
    }
    
    // Apply softmax function row-wise.
    Matrix softmax() const {
        Matrix result(rows, cols);
        // For each row in the matrix, compute softmax across the columns.
        for (int i = 0; i < rows; i++){
            // To improve numerical stability, find the maximum value in the row.
            double max_val = data[i][0];
            for (int j = 1; j < cols; j++){
                if (data[i][j] > max_val) {
                    max_val = data[i][j];
                }
            }
            double sum = 0.0;
            vector<double> exps(cols);
            // Compute exponentials for each element.
            for (int j = 0; j < cols; j++){
                exps[j] = exp(data[i][j] - max_val);
                sum += exps[j];
            }
            // Normalize the exponentials to get probabilities.
            for (int j = 0; j < cols; j++){
                result.data[i][j] = exps[j] / sum;
            }
        }
        return result;
    }
    
    // Transpose the matrix.
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
};

// ------------------------- Linear Layer -------------------------
// This class implements a simple fully-connected (dense) layer.
// It holds a weight matrix and a bias vector. The forward function
// computes: output = input * weights + bias.
class Linear {
public:
    Matrix weights;
    Matrix bias;
    
    // Constructor: input_dim is the dimension of input,
    // output_dim is the dimension of output.
    Linear(int input_dim, int output_dim) : weights(input_dim, output_dim), bias(1, output_dim) {
        weights.randomize();
        bias.randomize();
    }
    
    // Forward pass for the linear layer.
    // 'input' is assumed to be of shape (batch_size, input_dim)
    Matrix forward(const Matrix &input) {
        Matrix output = input.dot(weights); // Multiply input with weights.
        // Add bias to each row of the output.
        for (int i = 0; i < output.rows; i++){
            for (int j = 0; j < output.cols; j++){
                output.data[i][j] += bias.data[0][j];
            }
        }
        return output;
    }
};

// ------------------------- Self-Attention Layer -------------------------
// This class implements a single-head self-attention mechanism.
// It creates four linear layers: three for generating queries, keys,
// and values, and one for the final output projection.
class SelfAttention {
public:
    int model_dim;
    double scale;
    Linear Wq;
    Linear Wk;
    Linear Wv;
    Linear Wo;
    
    // Constructor: model_dim is the dimension of the model.
    SelfAttention(int model_dim) 
        : model_dim(model_dim),
          // Scale factor is the square root of model dimension for normalization.
          scale(sqrt(model_dim)),
          Wq(model_dim, model_dim),
          Wk(model_dim, model_dim),
          Wv(model_dim, model_dim),
          Wo(model_dim, model_dim) {}
    
    // Forward pass for self-attention.
    // 'input' is assumed to be a matrix of shape (sequence_length, model_dim)
    Matrix forward(const Matrix &input) {
        // Compute queries, keys, and values using linear layers.
        Matrix Q = Wq.forward(input); // Queries: (seq_len, model_dim)
        Matrix K = Wk.forward(input); // Keys: (seq_len, model_dim)
        Matrix V = Wv.forward(input); // Values: (seq_len, model_dim)
        
        // Compute the raw attention scores: Q * K^T (dot product between queries and keys).
        Matrix K_transposed = K.transpose(); // Transpose keys to shape (model_dim, seq_len)
        Matrix scores = Q.dot(K_transposed);   // (seq_len, seq_len)
        
        // Scale the scores by dividing by sqrt(model_dim) to help with gradients.
        for (int i = 0; i < scores.rows; i++){
            for (int j = 0; j < scores.cols; j++){
                scores.data[i][j] /= scale;
            }
        }
        
        // Apply softmax to the scores row-wise to obtain attention weights.
        Matrix attention = scores.softmax(); // (seq_len, seq_len)
        
        // Multiply the attention weights by V to get the context vectors.
        Matrix context = attention.dot(V); // (seq_len, model_dim)
        
        // Apply a final linear projection.
        Matrix output = Wo.forward(context); // (seq_len, model_dim)
        
        return output;
    }
};

// ------------------------- Feed-Forward Network -------------------------
// This class implements the position-wise feed-forward network.
// It uses two linear layers with a ReLU activation in between.
class FeedForward {
public:
    Linear linear1;
    Linear linear2;
    
    // Constructor: model_dim is the input and output dimension,
    // hidden_dim is the dimension of the hidden layer.
    FeedForward(int model_dim, int hidden_dim)
        : linear1(model_dim, hidden_dim), linear2(hidden_dim, model_dim) {}
    
    // Forward pass: first apply the first linear layer, then ReLU activation,
    // and finally the second linear layer.
    Matrix forward(const Matrix &input) {
        Matrix hidden = linear1.forward(input); // (seq_len, hidden_dim)
        hidden = hidden.relu();                 // Apply ReLU activation element-wise.
        Matrix output = linear2.forward(hidden);  // (seq_len, model_dim)
        return output;
    }
};

// ------------------------- Transformer Block -------------------------
// This class combines the self-attention layer and feed-forward network
// with residual (skip) connections.
class TransformerBlock {
public:
    SelfAttention self_attention;
    FeedForward feed_forward;
    
    // Constructor: model_dim is the model dimension and hidden_dim is used in feed-forward.
    TransformerBlock(int model_dim, int hidden_dim) 
        : self_attention(model_dim), feed_forward(model_dim, hidden_dim) {}
    
    // Helper function: element-wise addition of two matrices.
    Matrix addMatrices(const Matrix &A, const Matrix &B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            cout << "Matrix dimensions do not match for residual connection." << endl;
            exit(1);
        }
        Matrix result(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++){
            for (int j = 0; j < A.cols; j++){
                result.data[i][j] = A.data[i][j] + B.data[i][j];
            }
        }
        return result;
    }
    
    // Forward pass for the transformer block.
    // It applies self-attention, adds a residual connection,
    // then applies the feed-forward network, and adds another residual connection.
    Matrix forward(const Matrix &input) {
        // --- Self-Attention Sublayer ---
        Matrix attention_output = self_attention.forward(input);
        // Add the input (residual connection) to the self-attention output.
        Matrix attention_residual = addMatrices(input, attention_output);
        
        // --- Feed-Forward Sublayer ---
        Matrix feedforward_output = feed_forward.forward(attention_residual);
        // Add the residual connection.
        Matrix output = addMatrices(attention_residual, feedforward_output);
        
        return output;
    }
};

// ------------------------- Main Function -------------------------
int main() {
    // Seed the random number generator for reproducibility.
    srand(time(0));
    
    // Define dimensions for the synthetic data and transformer model.
    int sequence_length = 5; // Number of tokens (or time steps) in the sequence.
    int model_dim = 8;       // Dimension of each token's embedding.
    int hidden_dim = 16;     // Dimension of the hidden layer in the feed-forward network.
    
    // Create synthetic input data: a matrix of shape (sequence_length, model_dim)
    Matrix input(sequence_length, model_dim);
    input.randomize();  // Fill the matrix with random values in the range [-1, 1].
    
    // Display the synthetic input data.
    cout << "Synthetic Input Data:" << endl;
    input.print();
    cout << endl;
    
    // Create a transformer block instance with the specified dimensions.
    TransformerBlock transformer(model_dim, hidden_dim);
    
    // Perform a forward pass through the transformer block.
    Matrix output = transformer.forward(input);
    
    // Display the output from the transformer block.
    cout << "Transformer Output Data:" << endl;
    output.print();
    
    return 0;
}
