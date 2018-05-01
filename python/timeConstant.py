% Simulations to verify analytical estimates of error dependencies on network time constants 

from ca.nengo.math import Function
from ca.nengo.math.impl import FourierFunction
from ca.nengo.math.impl import IndicatorPDF
from ca.nengo.math.impl import ConstantFunction
from ca.nengo.model import SimulationMode
from ca.nengo.plot import Plotter
from ca.nengo.util import MU
import math

networks = [interneuron, dualTC, adapting, depressing, butterworth, interneuronFeedback]

tau = [.005, .01, .05, .1, .2, .5]
signalBandwidth = 15
frequencies = MU.makeVector(.1, .1, signalBandwidth)
componentRMS = math.sqrt(1.0 / len(frequencies)); 
signal = FourierFunction(frequencies, MU.uniform(1, len(frequencies), componentRMS/.707)[0], MU.random(1, len(frequencies), IndicatorPDF(-.5, .5))[0])

noiseBandwidth = 500

for network in networks:
	network.setMode(SimulationMode.DIRECT);
	network.setStepSize(.0005);	
	signalPower = []
	noisePower = []
	
	for t in tau:
		network.setTau(t)
		
		network.setInputFunction(signal);
		network.clearErrors();
		network.reset(0)
		network.run(0, 10)
		signalPower.append(MU.variance(MU.prod(network.getOutputData().getValues(), [1]), 0))
		
		network.setInputFunction(ConstantFunction(1, 0));
		network.setNoise(1000, 1000);
		network.reset(0)
		network.run(0, 10);
		network.clearErrors();
		noisePower.append(MU.variance(MU.prod(network.getOutputData().getValues(), [1]), 0))

	Plotter.plot(tau, signalPower, "%s signal power" %network.getName());
	Plotter.plot(tau, noisePower, "%s noise power" %network.getName());
	network.setStepSize(.001);
	