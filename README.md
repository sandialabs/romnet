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
	(e.g. python3 RomNet.py ../input/MassSpringDamper/DeepONet/)





--------------------------------------------------------------------------------------
## Test Cases:

Please, refer to the presentations in $WORKSPACE_PATH/ROMNet/romnet/docs/test_cases for info and running instructions


List of Implemented Test Cases (Note: it is highly recommended to run the test cases in the suggested order):

	- 1. Mass-Spring-Damper System
	
		- MSD_TestCase1:    Fully data-driven training of Vanilla DeepONet
		- MSD_TestCase2:    Physics-informed training of Vanilla DeepONet
		- MSD_TestCase3:    (Parts 1 to 5) Fully data-driven training of SVD-DeepONet 
		- MSD_TestCase4:    (Parts 1 to 4) Fully data-driven training of shared-trunk SVD-DeepONet 
		- MSD_TestCase5:    Probabilistic Vanilla DeepONet via MC Dropout
		- MSD_TestCase6:    Probabilistic Vanilla DeepONet via Variational Inference with deterministic parameters and fixed Likelihood's SD
		- MSD_TestCase7:    Probabilistic Vanilla DeepONet via Variational Inference with deterministic parameters and calibrated Likelihood's SD
		- MSD_TestCase8:    Probabilistic Vanilla DeepONet via Variational Inference with random parameters (i.e., TFP layers) and fixed Likelihood's SD
		- MSD_TestCase9:    Probabilistic Vanilla DeepONet via Variational Inference with random parameters (i.e., TFP layers) and calibrated Likelihood's SD
		
	
	- 2. Translating Hyperbolic Function
		
		- TransTanh_TestCase1: Fully data-driven training of Vanilla DeepONet
		- TransTanh_TestCase2: Physics-informed training of flexDeepONet
		- TransTanh_TestCase3: Fully data-driven training of Vanilla DeepONet
		- TransTanh_TestCase4: Physics-informed training of flexDeepONet
	
	
	- 3. 0D Isobaric Reactor 
	
		- 0DReact_*_TestCase1: Fully data-driven training of Vanilla DeepONet in the reduced-order (i.e., principal components) space
		- 0DReact_*_TestCase2: Fully data-driven training of flexDeepONet in the reduced-order (i.e., principal components) space
		- 0DReact_*_TestCase3: Fully data-driven training of Vanilla DeepONet in the original (i.e., thermodynamic state variables) space
		- 0DReact_*_TestCase4: Fully data-driven training of flexDeepONet in the original (i.e., thermodynamic state variables) space
		- 0DReact_*_TestCase5: -
		- 0DReact_*_TestCase6: Physics-informed training of flexDeepONet in the original (i.e., thermodynamic state variables) space
		- 0DReact_*_TestCase7: -
		- 0DReact_*_TestCase8: -
		- 0DReact_*_TestCase9: -
		- 0DReact_*_TestCase10: Test case for comparing flexDeepONet to flexMIONet
		- 0DReact_*_TestCase11: Frozen trunks (precomputed NNs from scenario-aggregated dimensionality reductions) DeepONet
		- 0DReact_*_TestCase12: NonLinear-DeepONet: flexDeepONet with a FNN decoder replacing the dot-product layer
		
	- 4. Moving Rectangle
	
		- Rect_TestCase1:  Fully data-driven training of vanilla DeepONet for a rotating-translating-scaling rigid body (~ rectangle)
		- Rect_TestCase2: -
		- Rect_TestCase3: -
		- Rect_TestCase4: -
		- Rect_TestCase5: Fully data-driven training of flexDeepONet for a rotating-translating-scaling rigid body (~ rectangle)
		- Rect_TestCase6: - 
		- Rect_TestCase7: - 
		
		
	- 5. Evolving PDF
		
		- PDFEvolve_TestCase1: Evolving Multimodal PDFs binned in 50 groups 



--------------------------------------------------------------------------------------
## Implemented NN Algorithms:

- Deterministic Neural Networks and DeepONets

- Probabilistic Neural Networks and DeepONets

	- Monte Carlo Dropout

	- Bayes by Backprop