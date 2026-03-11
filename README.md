# hybrid_quantum_classical_models
Python implementation of hybrid quantum classical models, HQResNet and HQCNN.


# Required Python libraries
Install all the required Python libraries.

pip install matplotlib

pip install torch

pip install torchvision

pip install qiskit

pip install qiskit_machine_learning

Note: matplotlib is for plotting all figures, and torch is for building classical neural networks and conneting them together with quantum parts,
while qiskit and qiskit_machine_learning are for creating quantum neural networks as well as working as the quantum environment simulator.


# Dataset Access
To get the dataset used in this work, please visit below link:

https://drive.google.com/file/d/1UhzYuK8zP107jMIWKcWn1-2r2tMyogDj/view?usp=sharing

Note: This dataset is a part of PlantVillage, the source is: https://github.com/spMohanty/PlantVillage-Dataset


# Python file instruction
1) CNN.py is the implementation of a classical CNN model;
2) ViT.py is the code for a Vision Transformer;
3) ResNet152.py shows the code for a ResNet-152 model;
4) HQCNN is the implementation of the hybrid Quantum Convolutional Neural Network model;
5) HQResNet presents the proposed hybrid quantum classical neural network architecture based on ResNet-152. 


# Process to run
1) Install above Python libraries;
2) Download above dataset and unzip it;
3) Download .py files and add them to your Python IDE project;
4) Change variable data_dir to the path of the dataset on your computer;
5) Execute the Python code.
   
