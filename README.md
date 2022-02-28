# Neural-Network-Based Surrogates for Reduced Order Model Dynamics (ROMNet)



--------------------------------------------------------------------------------------
## Executing RomNET:

1. In $WORKSPACE_PATH, create a ROMNET folder

2. Inside $WORKSPACE_PATH/ROMNet/, clone the ROMNet repository and rename it "romnet"

	Note: $WORKSPACE_PATH
					├── ...
					├── ROMNET
					│		└── romnet
									├── app
									├── database
									├── ...

3. From $WORKSPACE_PATH/ROMNet/romnet/, install the code (i.e., $ python3 setup.py install)

4. From $WORKSPACE_PATH/ROMNet/romnet/app/, launch the code (i.e., $ python3 RomNet.py path-to-input-folder) 
	(e.g. fpython3 RomNet.py ../input/MassSpringDamper/DeepONet/)





--------------------------------------------------------------------------------------
## Test Cases:

Please, refer to the presentations in $WORKSPACE_PATH/ROMNet/romnet/docs/test_cases for info and running instructions

It is highly recommended to run the set test cases in the following order:
	
	- 1. Mass-Spring-Damper System
	
	- 2. Translating Hyperbolic Function
	
	- 3. 0D Isobaric Reactor 




--------------------------------------------------------------------------------------
## Implemented NN Algorithms:

- Deterministic Neural Networks

- Probabilistic Neural Networks

	- Monte Carlo Dropout

	- Bayes by Backprop