/*
                                                                           
   (               )                                        )              
 ( )\     )     ( /(       (                  (  (     ) ( /((             
 )((_) ( /(  (  )\())`  )  )(   (  `  )   (   )\))( ( /( )\())\  (   (     
((_)_  )(_)) )\((_)\ /(/( (()\  )\ /(/(   )\ ((_))\ )(_)|_))((_) )\  )\ )  
 | _ )((_)_ ((_) |(_|(_)_\ ((_)((_|(_)_\ ((_) (()(_|(_)_| |_ (_)((_)_(_/(  
 | _ \/ _` / _|| / /| '_ \) '_/ _ \ '_ \/ _` |/ _` |/ _` |  _|| / _ \ ' \)) 
 |___/\__,_\__||_\_\| .__/|_| \___/ .__/\__,_|\__, |\__,_|\__||_\___/_||_|  
                    |_|           |_|         |___/                         

 For more information on back-propagation refer to:
 Chapter 18 of Russel and Norvig (2010).
 Artificial Intelligence - A Modern Approach.
 */

#include "CNeuralNet.h"

Node::Node(int NumInputs) : m_NumInputs(NumInputs)
{
	for (int i = 0; i < NumInputs; i++)
	{
		m_vecWeight.push_back(0.0f);
	}
}

/**
 The constructor of the neural network. This constructor will allocate memory
 for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 and output layers.
*/
CNeuralNet::CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff) :
cutoff(mse_cutoff), learnRate(lRate)
	//you probably want to use an initializer list here
{
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		hiddenLayer.push_back(Node(inputLayerSize));
	}
	for (int i = 0; i < outputLayerSize; i++)
	{
		outputLayer.push_back(Node(hiddenLayerSize));
	}
	initWeights();
}
/**
 The destructor of the class. All allocated memory will be released here
*/
CNeuralNet::~CNeuralNet() {
	delete[] inputPair;
	hiddenLayer.clear();
	outputLayer.clear();
	inputLayer.clear();
	sig_out.clear();
	hiddenInputs.clear();
}
/**
 Method to initialize the both layers of weights to random numbers
*/
void CNeuralNet::initWeights(){

	for (int i = 0; i < hiddenLayer.size(); i++)
	{
		for (int j = 0; j < hiddenLayer[i].m_NumInputs; j++)
		{
			hiddenLayer[i].m_vecWeight[j] = RandomClamped();
		}
	}

	for (int i = 0; i < outputLayer.size(); i++)
	{
		for (int j = 0; j < outputLayer[i].m_NumInputs; j++)
		{
			outputLayer[i].m_vecWeight[j] = RandomClamped();
		}
	}
}
/**
 This is the forward feeding part of back propagation.
 1. This should take the input and copy the memory (use memcpy / std::copy)
 to the allocated _input array.
 2. Compute the output of at the hidden layer nodes 
 (each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
 3. Repeat step 2, but this time compute the output at the output layer
*/
void CNeuralNet::feedForward(const double * const inputs) {
	inputPair = new double[2];
	sig_out.clear();
	std::copy(inputs,inputs+2, inputPair);
	double totalInput = 0;
	for (int i = 0; i < hiddenLayer.size(); i++)
	{
		totalInput = 0;
		for (int j = 0; j < hiddenLayer[i].m_NumInputs; j++)
		{	
			totalInput += hiddenLayer[i].m_vecWeight[j] * inputPair[j];		
		}
		sig_out.push_back(Sigmoid(totalInput));
	}
	hiddenInputs.clear();
	hiddenInputs = sig_out;
	sig_out.clear();
	for (int i = 0; i < outputLayer.size(); i++)
	{
		totalInput = 0;
		for (int j = 0; j < outputLayer[i].m_NumInputs; j++)
		{
			totalInput += outputLayer[i].m_vecWeight[j] * hiddenInputs[j];
		}
		sig_out.push_back(Sigmoid(totalInput));
	}
}
/**
 This is the actual back propagation part of the back propagation algorithm
 It should be executed after feeding forward. Given a vector of desired outputs
 we compute the error at the hidden and output layers (allocate some memory for this) and
 assign 'blame' for any error to all the nodes that fed into the current node, based on the
 weight of the connection.
 Steps:
 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs)
    for each output
 2. Compute the error at the hidden layer: sigmoid_d(hidden) * 
	sum(weights_o_h * difference between expected output and computed output at output layer)
	for each hidden layer node
 3. Adjust the weights from the hidden to the output layer: learning rate * error at the output layer * error at the hidden layer
    for each connection between the hidden and output layers
 4. Adjust the weights from the input to the hidden layer: learning rate * error at the hidden layer * input layer node value
    for each connection between the input and hidden layers
 5. REMEMBER TO FREE ANY ALLOCATED MEMORY WHEN YOU'RE DONE (or use std::vector ;)
*/
void CNeuralNet::propagateErrorBackward(const double * const desiredOutput){
	std::vector<double> error_output;
	std::vector<double> error_hidden;
	for (int i = 0; i < sig_out.size(); i++)
	{
		
		error_output.push_back(Sigmoid_d(sig_out[i])*(desiredOutput[i] - sig_out[i]));
		double total = 0;
		for (int j = 0; j < hiddenInputs.size(); j++)
		{
			total += (outputLayer[i].m_vecWeight[j] * (desiredOutput[i] - sig_out[i]));
			
			
		}		
	}

	for (int j = 0; j < hiddenInputs.size(); j++)
	{
		error_hidden.push_back(Sigmoid_d(hiddenInputs[j])*((error_output[0] * outputLayer[0].m_vecWeight[j]) + (error_output[1] * outputLayer[1].m_vecWeight[j])));
	}


	for (int j = 0; j < 150;j++)
	{
		outputLayer[0].m_vecWeight[j] += learnRate * error_output[0] * hiddenInputs[j];
		outputLayer[1].m_vecWeight[j] += learnRate * error_output[1] * hiddenInputs[j];
		hiddenLayer[j].m_vecWeight[0] += learnRate * error_hidden[j] * inputPair[0];
		hiddenLayer[j].m_vecWeight[1] += learnRate * error_hidden[j] * inputPair[1];		
	}
}
/**
This computes the mean squared error
A very handy formula to test numeric output with. You may want to commit this one to memory
*/
double CNeuralNet::meanSquaredError(const double * const desiredOutput){
	/*TODO:
	sum <- 0
	for i in 0...outputLayerSize -1 do
		err <- desiredoutput[i] - actualoutput[i]
		sum <- sum + err*err
	return sum / outputLayerSize
	*/
	double sum = 0;
	for (int i = 0; i < outputLayer.size(); i++)
	{
		double err = desiredOutput[i] - sig_out[i];
		sum += err*err;
	}
	double ret = sum / outputLayer.size();
	return (ret);
}
/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
for each training pattern:
  feed forward
  propagate backward
until the MSE becomes suitably small
*/
void CNeuralNet::train(const double** const inputs,
	const double** const outputs, uint trainingSetSize) {
	double mse = 0;
	for (int i = 0; i < trainingSetSize; i++)
	{
		feedForward(inputs[i]);
		propagateErrorBackward(outputs[i]);
		mse = meanSquaredError(outputs[i]);
		//printf("mse... %f \n", mse);
		if (mse < cutoff)
		{
			printf("mse... %i \n", i);
			break;
		}
	}

}
/**
Once our network is trained we can simply feed it some input though the feed forward
method and take the maximum value as the classification
*/
uint CNeuralNet::classify(const double * const input){
	feedForward(input);
	if (sig_out[0] > sig_out[1])
	{
		return 0;
	}
	else
	{
		return 1;
	}
}
/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(uint index) const{
	return 0; //TODO: fix me
}

double CNeuralNet::Sigmoid(double activation)
{
	return (1 / (1 + exp(-activation)));
}

double CNeuralNet::Sigmoid_d(double sigmoid_value)
{
	double value = sigmoid_value*(1 - sigmoid_value);
	
	return (value);
}