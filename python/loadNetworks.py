# Loads all of the network models 

from com.bptripp.diff import *

interneuron = InterneuronNetwork(.1, 2000)
dualTC = DualTCNetwork(.005, .1, 1)
interneuronFeedback = FeedbackNetwork([1000, 1000], .1, [[-5, -5], [5, -15]], [[10], [30]], [[10, 0]]) #sinusoidal input
interneuronFeedback.setName("interneuronFeedback")
interneuronFeedbackR = FeedbackNetwork([1000, 1000], .1, [[-5, -7.5], [3.3333, -15]], [[10], [20]], [[10, 0]]) #ramp input
interneuronFeedbackR.setName("interneuronFeedback")
butterworth = FeedbackNetwork([1000, 1000], .1, [[-8.8858, 8.8858], [-8.8858, -8.8858]], [[27.4892], [-27.4892]], [[5.7446, 0]]) #sinusoidal input
butterworth.setName("butterworth")
butterworthR = FeedbackNetwork([1000, 1000], .1, [[-8.8858, 19.9931], [-3.9492, -8.8858]], [[27.4892], [-12.2174]], [[5.7446, 0]]) #ramp input
butterworthR.setName("butterworth")
adapting = AdaptingNetwork(1600, 400, .005)
adapting.setTau(.1)
depressing = DepressionNetwork(2000)
depressing.setTau(.1)
networks = [interneuron, dualTC, adapting, depressing, butterworth, interneuronFeedback, butterworthR, interneuronFeedbackR]

#dualTCF = DualTCNetwork(.005, .015, 1) #variant on dualTC with faster time constant to make errors clearer 
#dualTCF.setName("dualTCF")
#dualTCU = DualTCNetwork(.005, .015, 0) #variant on dualTC with uncorrelated errors, faster time constant
#dualTCU.setName("dualTCU")

for network in networks:
	world.add(network)
