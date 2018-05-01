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

public class DualTCNetwork extends DifferentiatorNetwork {

	private static final long serialVersionUID = 1L;
	
	private static final String DIRECT = "direct";
	private static final String DELAYED = "delayed";
	
	private Projection myDirectProjection;
	private Projection myDelayedProjection;

	/**
	 * @param tauPSC Time constant of post-synaptic current decay in fast projection. 
	 * @param slowTauPSC Time constant of post-synaptic current decay in slow projection.
	 * @param correlatedError If true, errors in two projections are identical; if false they are uncorrelated
	 * 
	 * @throws StructuralException
	 */
	public DualTCNetwork(float tauPSC, float slowTauPSC, boolean correlatedError) throws StructuralException {
		setName("dualTC");

		float tauDifference = slowTauPSC - tauPSC;
		
		getInputEnsemble().addDecodedTermination("input", MU.I(1), TAU_IO, false);
		addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), getInputEnsemble().getTermination("input"));
		
		NEFEnsemble output = getOutputEnsemble();
		output.addDecodedTermination("direct", new float[][]{new float[]{1f / tauDifference}}, tauPSC, false);
		output.addDecodedTermination("delayed", new float[][]{new float[]{-1f / tauDifference}}, slowTauPSC, false);
		myDirectProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), output.getTermination("direct"));
		if (correlatedError) {
			myDelayedProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), output.getTermination("delayed"));
		} else {			
			int n = getInputEnsemble().getNodes().length;
			NEFEnsemble uncorrelated = myEnsembleFactory.make("input2", n, 1, "diff_input2_"+n, false);
			uncorrelated.addDecodedTermination("input", MU.I(1), TAU_IO, false);			
			addNode(uncorrelated);
			addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), uncorrelated.getTermination("input"));
			myDelayedProjection = addProjection(uncorrelated.getOrigin(NEFEnsemble.X), output.getTermination("delayed"));
		}
		
	}
	
	@Override
	public void clearErrors() {
		((Noisy) myDirectProjection.getOrigin()).setNoise(new NoiseFactory.NoiseImplNull());
		((Noisy) myDelayedProjection.getOrigin()).setNoise(new NoiseFactory.NoiseImplNull());
	}

	@Override
	public void setDistortion(int nInput, int nDiff) {
		((Noisy) myDirectProjection.getOrigin()).setNoise(makeDistortion(nInput));
		((Noisy) myDelayedProjection.getOrigin()).setNoise(makeDistortion(nInput));
	}

	@Override
	public void setNoise(int nInput, int nDiff) {
		((Noisy) myDirectProjection.getOrigin()).setNoise(makeNoise(nInput));
		((Noisy) myDelayedProjection.getOrigin()).setNoise(makeNoise(nInput));
	}

	@Override
	public void setTau(float tau) {
		try {
			getOutputEnsemble().getTermination(DELAYED).setTau(tau);
			float tauDifference = tau - getOutputEnsemble().getTermination(DIRECT).getTau();
			((DecodedTermination) getOutputEnsemble().getTermination(DIRECT)).setTransform(new float[][]{new float[]{1f / tauDifference}});
			((DecodedTermination) getOutputEnsemble().getTermination(DELAYED)).setTransform(new float[][]{new float[]{-1f / tauDifference}});
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void disableParisien() {
		myDirectProjection.removeBias();
		myDelayedProjection.removeBias();		
	}

	@Override
	public void enableParisien(float propInhibitory) throws StructuralException {
		int n = Math.round(propInhibitory * (float) getOutputEnsemble().getNodes().length);
		enableParisien(myDirectProjection, n);
		enableParisien(myDelayedProjection, n);
	}

	public static void main(String[] args) throws StructuralException {
		new DualTCNetwork(.005f, .1f, false);
	}
}
