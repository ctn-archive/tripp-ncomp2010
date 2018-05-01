# Example simulations with ramp input

from com.bptripp.diff import *
from ca.nengo.model import SimulationMode
from java.io import File
from ca.nengo.math import PDFTools

ramp = Util.RAMP
networks = [interneuron, dualTC, adapting, depressing, butterworthR, interneuronFeedbackR]
for network in networks:
	network.enableParisien(.25)
	network.setMode(SimulationMode.DEFAULT)
	network.setInputFunction(ramp)
	network.setStepSize(.0005)	
	network.run(-.5, 3)
	network.exportAll(File("example_"+network.getName()+".mat"))

	network.setMode(SimulationMode.DIRECT)
	PDFTools.setSeed(1)
	network.run(-.5, 3)
	network.exportAll(File("example_"+network.getName()+"_direct.mat"))
	network.disableParisien()
