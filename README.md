# RNN
import numpy as np

# Sample text data
text = "this is a test"
words = text.split()

# Create a vocabulary
vocab = sorted(list(set(words)))
word_to_index = {word: index for index, word in enumerate(vocab)}
index_to_word = {index: word for index, word in enumerate(vocab)}
vocab_size = len(vocab)

# Create input and target sequences
input_sequences = []
target_sequences = []
for i in range(len(words) - 3):
    input_sequences.append([word_to_index[word] for word in words[i:i+3]])
    target_sequences.append(word_to_index[words[i+3]])

# Convert to numpy arrays
X = np.array(input_sequences)
y = np.array(target_sequences)

# Define RNN model parameters
input_size = vocab_size
hidden_size = 4
output_size = vocab_size

np.random.seed(42)
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def rnn_step(x, h_prev):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    y = np.dot(Why, h) + by
    return h, y

# Training
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    h = np.zeros((hidden_size, 1))
    loss = 0

    for i in range(len(X)):
        hs = {}  # to store hidden states
        hs[-1] = np.copy(h)

        # Forward
        for t in range(3):
            inputs = np.zeros((vocab_size, 1))
            inputs[X[i][t]] = 1
            h, y_pred = rnn_step(inputs, h)
            hs[t] = h

        probs = softmax(y_pred)
        target_index = y[i]
        loss += -np.log(probs[target_index])

        # Backward (only simple update)
        dy = probs
        dy[target_index] -= 1

        # Gradients
        dWhy = np.dot(dy, hs[2].T)
        dby = dy

        dh = np.dot(Why.T, dy) * (1 - hs[2] ** 2)

        dWxh = np.dot(dh, np.zeros((vocab_size, 1)).T)  
        dWhh = np.dot(dh, hs[1].T)
        dbh = dh

        # Update parameters
        Why -= learning_rate * dWhy
        by -= learning_rate * dby
        Whh -= learning_rate * dWhh
        bh -= learning_rate * dbh

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Make prediction
h = np.zeros((hidden_size, 1))
inputs = np.zeros((vocab_size, 1))
inputs[word_to_index['this']] = 1
h, _ = rnn_step(inputs, h)

inputs = np.zeros((vocab_size, 1))
inputs[word_to_index['is']] = 1
h, _ = rnn_step(inputs, h)

inputs = np.zeros((vocab_size, 1))
inputs[word_to_index['a']] = 1
h, y_pred = rnn_step(inputs, h)

predicted_index = np.argmax(softmax(y_pred))
predicted_word = index_to_word[predicted_index]
print(f"\nPredicted word: {predicted_word}")
