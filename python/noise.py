# Numerical simulations to validate analytical results on noise propagation

from com.bptripp.diff import *
from ca.nengo.math.impl import ConstantFunction
from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NoiseFactory
from ca.nengo.io import MatlabExporter
from ca.nengo.plot import Plotter
from ca.nengo.util import MU
from java.io import File
import math

nInput = range(200, 2001, 400)
nDiff = 1000;

networks = [interneuron, dualTC, adapting, depressing, butterworth, interneuronFeedback]

exporter = MatlabExporter()
for network in networks:
	network.setInputFunction(ConstantFunction(1, 0));
	network.setStepSize(.0001)
	network.setMode(SimulationMode.DIRECT);

	inputVariance = [];
	outputVariance = [];
	
	for n in nInput:
		network.setNoise(n, nDiff);
		#network.setDistortion(n, nDiff);
		network.reset(0)
		network.run(0, 10);
		inputVariance.append(MU.variance(MU.prod(network.getInputEnsembleData().getValues(), [1]), 0))
		outputVariance.append(MU.variance(MU.prod(network.getOutputData().getValues(), [1]), 0))
		
	network.clearErrors();
	Plotter.plot(nInput, outputVariance, "output")
	
exporter.write(File("noise.mat"));		
	