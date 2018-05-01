/*
 * Created on 1-Apr-08
 */
package com.bptripp.diff;

import ca.nengo.model.Projection;
import ca.nengo.model.StructuralException;
import ca.nengo.model.Noise.Noisy;
import ca.nengo.model.impl.FunctionInput;
import ca.nengo.model.impl.NoiseFactory;
import ca.nengo.model.nef.NEFEnsemble;
import ca.nengo.model.nef.impl.DecodedTermination;
import ca.nengo.util.MU;

/**
 * A DifferentiatorNetwork based on propagation delay via an intermediate neuron ensemble.
 *  
 * @author Bryan Tripp
 */
public class InterneuronNetwork extends DifferentiatorNetwork {

	private static final long serialVersionUID = 1L;
	
	private NEFEnsemble myInterneurons;
	private Projection myInputOutputProjection;
	private Projection myInputInterneuronProjection;
	private Projection myInterneuronOutputProjection;

	/**
	 * @param tauPSC Time constant of post-synaptic current decay in intermediate ensemble
	 * @param numInterneurons Number of neurons in intermediate ensemble
	 * 
	 * @throws StructuralException
	 */
	public InterneuronNetwork(float tauPSC, int numInterneurons) throws StructuralException {
		setName("interneuron");
		
		getInputEnsemble().addDecodedTermination("input", MU.I(1), TAU_IO, false);
		addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), getInputEnsemble().getTermination("input"));
		
		myInterneurons = myEnsembleFactory.make("interneurons", numInterneurons, 1, "diff_inter_"+numInterneurons, false);
		myInterneurons.addDecodedTermination("input", MU.I(1), tauPSC, false);
		myInputInterneuronProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), myInterneurons.getTermination("input"));
		addNode(myInterneurons);

		getOutputEnsemble().addDecodedTermination("direct", new float[][]{new float[]{1/tauPSC}}, tauPSC, false);
		getOutputEnsemble().addDecodedTermination("indirect", new float[][]{new float[]{-1/tauPSC}}, tauPSC, false);
		myInputOutputProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), getOutputEnsemble().getTermination("direct"));
		myInterneuronOutputProjection = addProjection(myInterneurons.getOrigin(NEFEnsemble.X), getOutputEnsemble().getTermination("indirect"));
	}
	
	@Override
	public void disableParisien() {
		myInputOutputProjection.removeBias();
		myInputInterneuronProjection.removeBias();
		myInterneuronOutputProjection.removeBias();
	}

	@Override
	public void enableParisien(float propInhibitory) throws StructuralException {
		int nOutput = Math.round(propInhibitory * (float) getOutputEnsemble().getNodes().length);
		int nInterneurons = Math.round(propInhibitory * (float) myInterneurons.getNodes().length);
		enableParisien(myInputOutputProjection, nOutput);
		enableParisien(myInputInterneuronProjection, nInterneurons);
		enableParisien(myInterneuronOutputProjection, nOutput);
	}

	@Override
	public void clearErrors() {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
			((Noisy) myInterneurons.getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}


	@Override
	public void setDistortion(int nInput, int nDiff) {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nInput));
			((Noisy) myInterneurons.getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nDiff));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}


	@Override
	public void setNoise(int nInput, int nDiff) {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nInput));
			((Noisy) myInterneurons.getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nDiff));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}


	@Override
	public void setTau(float tau) {
		try {
			myInterneurons.getTermination("input").setTau(tau);
			
			DecodedTermination direct = (DecodedTermination) getOutputEnsemble().getTermination("direct");
			direct.setTau(tau);
			direct.setTransform(new float[][]{new float[]{1/tau}});
			
			DecodedTermination indirect = (DecodedTermination) getOutputEnsemble().getTermination("indirect");
			indirect.setTau(tau);
			indirect.setTransform(new float[][]{new float[]{-1/tau}});
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}

	}

}
