import numpy as np

# Fonction d'activation (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction sigmoid pour la rétropropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialisation des paramètres
input_layer_neurons = 2  # Nombre de neurones dans la couche d'entrée
hidden_layer_neurons = 15  # Nombre de neurones dans la couche cachée
output_neurons = 1  # Nombre de neurones dans la couche de sortie

# Entrées d'entraînement
X = np.array([[0,0], [0,1], [1,0], [1,1]])
# Sorties d'entraînement
y = np.array([[0], [1], [1], [0]])

# Poids et biais aléatoires
hidden_weights = np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1,hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons,output_neurons))
output_bias = np.random.uniform(size=(1,output_neurons))

# Taux d'apprentissage
lr = 0.1

# Entraînement
for epoch in range(10000):
    # Propagation avant
    hidden_layer_activation = np.dot(X,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Calcul de l'erreur
    error = y - predicted_output
    
    # Rétropropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Mise à jour des poids et biais
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights += X.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

# Affichage des résultats
print("Poids après entraînement")
print(hidden_weights)
print("Poids de sortie")
print(output_weights)

print("Sorties prévues : ")
print(predicted_output)
