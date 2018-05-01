# Simulations with sinusoidal input to test frequency responses

from com.bptripp.diff import *
from ca.nengo.math.impl import SineFunction
from ca.nengo.model import SimulationMode
from ca.nengo.io import MatlabExporter
from java.io import File
import math

#the following networks must be loaded before running this script
networks = [interneuron, dualTC, adapting, depressing, butterworth, interneuronFeedback]
frequencies = [0.25, 0.4504, 0.8115, 1.4620, 2.6340, 4.7456, 8.5499, 15.4039, 27.7524, 50.0000]

for network in networks:
	network.enableParisien(.25)
	network.setStepSize(.0005)

	network.setMode(SimulationMode.CONSTANT_RATE)
	if network.getName() == "adapting" : 
		network.getNode("adapting").setMode(SimulationMode.RATE)
	
	maxAmplitude = 0.5
	if network.getName() == "butterworth" :
		maxAmplitude = 1.0/3.0
	if network.getName() == "adapting" :
		maxAmplitude = 1.0
	if network.getName() == "depression" :
		maxAmplitude = 1.0
		 
	exporter = MatlabExporter()
	exporter.add("frequencies", [frequencies])
	for i in range(len(frequencies)):
		frequency = frequencies[i]
		print "Network: ", network.getName(), " Frequency: ", frequency, "Hz"
		
		angularFrequency = 2 * math.pi * frequency
		amplification = angularFrequency
		if network.getName() == "depression" : 
			amplification = min(amplification, 10.0)
		amplitude = min(maxAmplitude, 1.0/amplification)   #normalize so that input, output, and state magnitudes <= 1  
		
		network.setInputFunction(SineFunction(angularFrequency, amplitude))

		network.run(0, 1.0+5.0/frequency)
		exporter.add("in%i" %i, network.getInputData())
		exporter.add("out%i" %i, network.getOutputData())

	#export simulation results to a Matlab file
	exporter.write(File(network.getName()+"_frequency_parisien.mat"));
	
	network.setStepSize(.001)
	network.disableParisien()