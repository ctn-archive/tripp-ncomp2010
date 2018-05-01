/*
 * Created on 1-Apr-08
 */
package com.bptripp.diff;

import ca.nengo.math.Function;
import ca.nengo.math.impl.IndicatorPDF;
import ca.nengo.model.Node;
import ca.nengo.model.Origin;
import ca.nengo.model.Projection;
import ca.nengo.model.RealOutput;
import ca.nengo.model.SimulationException;
import ca.nengo.model.SimulationMode;
import ca.nengo.model.StructuralException;
import ca.nengo.model.Noise.Noisy;
import ca.nengo.model.impl.FunctionInput;
import ca.nengo.model.impl.NoiseFactory;
import ca.nengo.model.nef.NEFEnsemble;
import ca.nengo.model.nef.NEFEnsembleFactory;
import ca.nengo.model.nef.impl.DecodedTermination;
import ca.nengo.model.nef.impl.NEFEnsembleFactoryImpl;
import ca.nengo.model.neuron.Neuron;
import ca.nengo.model.neuron.impl.ALIFNeuronFactory;
import ca.nengo.model.neuron.impl.ALIFSpikeGenerator;
import ca.nengo.model.neuron.impl.SpikingNeuron;
import ca.nengo.util.MU;

/**
 * A differentiator network in which differentiation is achieved through 
 * adapting LIF neurons. 
 * 
 * @author Bryan Tripp
 */
public class AdaptingNetwork extends DifferentiatorNetwork {

	private static final long serialVersionUID = 1L;
	
	private static final String COMPENSATING = "compensating";
	private static final String ADAPTING = "adapting";
	
	private NEFEnsemble myAdapting;
	private NEFEnsemble myCompensating;
	private float myPropAdapting;
	
	private Projection myInputAdaptingProjection;
	private Projection myInputCompensatingProjection;
	private Projection myAdaptingOutputProjection;
	private Projection myCompensatingOutputProjection;

	/**
	 * @param nAdapting Number of adapting neurons 
	 * @param nCompensating Number of non-adapting neurons that compensate for non-zero adapted activity  
	 * @param tauPSC Time constant of post-synaptic current within adapting and compensating neurons
	 * 
	 * @throws StructuralException
	 */
	public AdaptingNetwork(int nAdapting, int nCompensating, float tauPSC) throws StructuralException {
		setName("adapting");
		myPropAdapting = (float) nAdapting / ((float) nAdapting + (float) nCompensating);
		
		getInputEnsemble().addDecodedTermination("input", MU.I(1), TAU_IO, false);
		addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), getInputEnsemble().getTermination("input"));
		
		NEFEnsembleFactory aef = getALIFEnsembleFactory();		
		myAdapting = aef.make("adapting", nAdapting, 1, "adapting_diff_"+nAdapting, false);
		addNode(myAdapting);
		
		NEFEnsembleFactory ef = new NEFEnsembleFactoryImpl();		
		myCompensating = ef.make("compensating", nCompensating, 1);
		myCompensating.addDecodedTermination("input", MU.I(1), tauPSC, false);
		addNode(myCompensating);
		
		myAdapting.addDecodedTermination("input", MU.I(1), tauPSC, false); //have to wait for bias compensation simulations
		
		NEFEnsemble output = getOutputEnsemble();
		float[][] scale = new float[][]{new float[]{15f}}; //this starting value roughly corresponds to the starting (non-uniform) time constant  
		output.addDecodedTermination(ADAPTING, scale, tauPSC, false);
		output.addDecodedTermination(COMPENSATING, scale, tauPSC, false);
		
		myInputAdaptingProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), myAdapting.getTermination("input"));
		myInputCompensatingProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), myCompensating.getTermination("input"));
		myAdaptingOutputProjection = addProjection(myAdapting.getOrigin(NEFEnsemble.X), output.getTermination(ADAPTING));
		
		setCompensation(.1f);		

		try {
			getSimulator().addProbe(myCompensating.getName(), COMPENSATING, true);
		} catch (SimulationException e) {
			throw new StructuralException(e);
		}
	}

	/*
	 * (non-Javadoc)
	 * @see ca.bpt.diff.DifferentiatorNetwork#enableParisien(float)
	 */
	public void enableParisien(float propInhibitory) throws StructuralException {
		int nAdapting = Math.round(propInhibitory * (float) myAdapting.getNodes().length); 
		int nCompensating = Math.round(propInhibitory * (float) myCompensating.getNodes().length);
		int nOutput = Math.round(propInhibitory * (float) getOutputEnsemble().getNodes().length);

		enableParisien(myInputAdaptingProjection, nAdapting);
		enableParisien(myInputCompensatingProjection, nCompensating);
		myAdaptingOutputProjection.addBias(nOutput, TAU_INTERNEURONS, myAdaptingOutputProjection.getTermination().getTau(), true, false);
		enableParisien(myCompensatingOutputProjection, nOutput);
	}
	
	/*
	 * (non-Javadoc)
	 * @see ca.bpt.diff.DifferentiatorNetwork#disableParisien()
	 */
	public void disableParisien() {
		myInputAdaptingProjection.removeBias();
		myInputCompensatingProjection.removeBias();
		myAdaptingOutputProjection.removeBias();
		myCompensatingOutputProjection.removeBias();
	}
	
	/**
	 * @return factory for NEFEnsembles composed of adapting LIF neurons
	 */
	public static NEFEnsembleFactory getALIFEnsembleFactory() {
		NEFEnsembleFactory result = new NEFEnsembleFactoryImpl();
		float incN = .05f;
		float tauN = .2f;
		result.setNodeFactory(new ALIFNeuronFactory(new IndicatorPDF(200, 400), new IndicatorPDF(-2.5f, -1.5f), new IndicatorPDF(incN), .0005f, .02f, tauN));
		return result;
	}
	
	@Override
	public void setTau(float tau) {
		Node[] neurons = myAdapting.getNodes();
		for (int i = 0; i < neurons.length; i++) {
			SpikingNeuron neuron = (SpikingNeuron) neurons[i];
			ALIFSpikeGenerator generator = (ALIFSpikeGenerator) neuron.getGenerator();
			
			float alpha = getSlope(neuron) / neuron.getScale();
			float b = neuron.getBias();
			float c = neuron.getScale();
			
			float tauN = tau/2 * (b/c + 1);
			float A_N = (1/tau - 1/tauN) / alpha;
			generator.setIncN(A_N);
			generator.setTauN(tauN);
		}

		try {
			setCompensation(tau);

			float[][] scale = new float[][]{new float[]{2.5f / tau}};
			((DecodedTermination) getOutputEnsemble().getTermination(ADAPTING)).setTransform(scale);
			((DecodedTermination) getOutputEnsemble().getTermination(COMPENSATING)).setTransform(scale);
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}		
	}
	
	private void setCompensation(float tau) throws StructuralException {
		try {
			Function f = Util.getBiasCompensation(myAdapting, NEFEnsemble.X, tau*8);
			Origin origin = myCompensating.addDecodedOrigin(COMPENSATING, new Function[]{f}, Neuron.AXON);
			
			for (Projection p : getProjections()) {
				if (p.getTermination() == getOutputEnsemble().getTermination(COMPENSATING)) {
					removeProjection(getOutputEnsemble().getTermination(COMPENSATING));					
				}
			}
			myCompensatingOutputProjection = addProjection(origin, getOutputEnsemble().getTermination(COMPENSATING));			
		} catch (SimulationException e) {
			throw new StructuralException(e);
		}
	}
	
	/**
	 * @param neuron A spiking neuron model
	 * @return mean derivative of spike rate wrt represented quantity, over the range [-1,1] (obtained by simulation) 
	 */
	public static float getSlope(SpikingNeuron neuron) {
		SimulationMode mode = neuron.getMode();
		float slope = 0;
		
		try {
			neuron.setMode(SimulationMode.CONSTANT_RATE);
			neuron.setRadialInput(-1);
			neuron.run(0, 0);
			RealOutput low = (RealOutput) neuron.getOrigin(Neuron.AXON).getValues();
			neuron.setRadialInput(1);
			neuron.run(0, 0);
			RealOutput high = (RealOutput) neuron.getOrigin(Neuron.AXON).getValues();
			slope = (high.getValues()[0] - low.getValues()[0]) / 2f; 
			neuron.setMode(mode);
		} catch (SimulationException e) {
			throw new RuntimeException(e);
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
		
		return slope;
	}

	@Override
	public void clearErrors() {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
			((Noisy) myAdapting.getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
			((Noisy) myCompensating.getOrigin(COMPENSATING)).setNoise(new NoiseFactory.NoiseImplNull());
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}		
	}

	@Override
	public void setDistortion(int nInput, int nDiff) {
		int nAdapting = Math.round(nDiff*myPropAdapting);
		int nCompensating = Math.round(nDiff*(1-myPropAdapting));
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nInput));
			((Noisy) myAdapting.getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nAdapting));
			((Noisy) myCompensating.getOrigin(COMPENSATING)).setNoise(makeDistortion(nCompensating));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void setNoise(int nInput, int nDiff) {
		int nAdapting = Math.round(nDiff*myPropAdapting);
		int nCompensating = Math.round(nDiff*(1-myPropAdapting));
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nInput));
			((Noisy) myAdapting.getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nAdapting));
			((Noisy) myCompensating.getOrigin(COMPENSATING)).setNoise(makeNoise(nCompensating));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args) throws StructuralException {
		AdaptingNetwork an = new AdaptingNetwork(200, 100, .01f);
		an.setTau(.1f);
	}

}
