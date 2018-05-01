/*
 * Created on 1-Apr-08
 */
package com.bptripp.diff;

import ca.nengo.math.impl.IndicatorPDF;
import ca.nengo.model.Projection;
import ca.nengo.model.StructuralException;
import ca.nengo.model.Noise.Noisy;
import ca.nengo.model.impl.FunctionInput;
import ca.nengo.model.impl.NoiseFactory;
import ca.nengo.model.nef.NEFEnsemble;
import ca.nengo.model.nef.NEFEnsembleFactory;
import ca.nengo.model.nef.impl.DecodedTermination;
import ca.nengo.model.nef.impl.NEFEnsembleFactoryImpl;
import ca.nengo.model.nef.impl.NEFEnsembleImpl;
import ca.nengo.model.neuron.impl.LIFNeuronFactory;
import ca.nengo.util.MU;
import ca.nengo.util.impl.RandomHypersphereVG;

/**
 * A differentiator network in which band-pass input-output behavior arises from feedack dynamics. 
 * 
 * @author Bryan Tripp
 */
public class FeedbackNetwork extends DifferentiatorNetwork {

	private static final long serialVersionUID = 1L;

	private static String INPUT = "input";
	private static String FEEDBACK = "feedback";

	private float[][] myA;
	private float[][] myB;
	private NEFEnsemble myDiff;
	private Projection myInputDiffProjection;
	private Projection myDiffDiffProjection;
	private Projection myDiffOutputProjection;
	
	public FeedbackNetwork(int[] numInterneurons, float tauPSC, float[][] A, float[][] B, float[][] C) throws StructuralException {
		myA = A;
		myB = B;
		
		setName("feedback");
						
		getInputEnsemble().addDecodedTermination("input", MU.I(1), TAU_IO, false);
		addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), getInputEnsemble().getTermination("input"));
		
		//2D differentiator ensemble with specified # neurons along each dim
		NEFEnsembleFactory ef = new NEFEnsembleFactoryImpl();
		DimensionRatioVG encoderFactory = new DimensionRatioVG(true, 1, 1);
		encoderFactory.setRatio(new float[]{numInterneurons[0], numInterneurons[1]});
		ef.setEncoderFactory(encoderFactory);
		ef.setNodeFactory(new LIFNeuronFactory(.02f, .0005f, new IndicatorPDF(200, 400), new IndicatorPDF(-1.2f, .95f)));
		
		int n = numInterneurons[0]+numInterneurons[1];
		myDiff = ef.make("diff", n, 2, "feedback_diff_"+numInterneurons[0]+"_"+numInterneurons[1], false);
		((NEFEnsembleImpl) myDiff).setEvalPoints(new RandomHypersphereVG(false, (float) Math.sqrt(2), 0).genVectors(300, 2));
		addNode(myDiff);
		
		getOutputEnsemble().addDecodedTermination("diff", C, TAU_IO, false);
		
		myDiff.addDecodedTermination(FEEDBACK, getA(tauPSC), tauPSC, false);
		myDiff.addDecodedTermination(INPUT, getB(tauPSC), tauPSC, false);

		myInputDiffProjection = addProjection(getInputEnsemble().getOrigin(NEFEnsemble.X), myDiff.getTermination("input"));
		myDiffDiffProjection = addProjection(myDiff.getOrigin(NEFEnsemble.X), myDiff.getTermination("feedback"));
		myDiffOutputProjection = addProjection(myDiff.getOrigin(NEFEnsemble.X), getOutputEnsemble().getTermination("diff"));
	}
		
	
	private float[][] getA(float tauPSC) {
		return MU.sum(MU.I(myA.length), MU.prod(myA, tauPSC));		
	}
	
	private float[][] getB(float tauPSC) {
		return MU.prod(myB, tauPSC);		
	}

	@Override
	public void disableParisien() {
		myInputDiffProjection.removeBias();
		myDiffDiffProjection.removeBias();
		myDiffOutputProjection.removeBias();
	}

	@Override
	public void enableParisien(float propInhibitory) throws StructuralException {
		int nDiff = Math.round(propInhibitory * (float) myDiff.getNodes().length);
		int nOut = Math.round(propInhibitory * (float) getOutputEnsemble().getNodes().length);
		enableParisien(myInputDiffProjection, nDiff);
		myDiffDiffProjection.addBias(nDiff, TAU_INTERNEURONS, myDiffDiffProjection.getTermination().getTau(), true, false);
		enableParisien(myDiffOutputProjection, nOut);
	}

	@Override
	public void clearErrors() {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
			((Noisy) myDiff.getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Note: sets equal error on each output of the differentiator ensemble
	 */
	@Override
	public void setDistortion(int nInput, int nDiff) {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nInput));
			((Noisy) myDiff.getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nDiff));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void setNoise(int nInput, int nDiff) {
		try {
			((Noisy) getInputEnsemble().getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nInput));
			((Noisy) myDiff.getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nDiff));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void setTau(float tau) {
		try {
			DecodedTermination input = (DecodedTermination) myDiff.getTermination(INPUT);
			input.setTau(tau);
			input.setTransform(getB(tau));
			DecodedTermination feedback = (DecodedTermination) myDiff.getTermination(FEEDBACK);
			feedback.setTau(tau);
			feedback.setTransform(getA(tau));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * Allows us to specify ratio of vectors generated along each dimension. 
	 * 
	 * @author Bryan Tripp
	 */
	public static class DimensionRatioVG extends RandomHypersphereVG {

		private static final long serialVersionUID = 1L;
		float[] myRatio;
		
		public DimensionRatioVG(boolean surface, float radius, float axisClusterFactor) {
			super(surface, radius, axisClusterFactor);
			myRatio = new float[]{1};
		}
		
		public void setRatio(float[] ratio) {
			myRatio = ratio;
		}
		
		public float[] getRatio() {
			return myRatio;
		}
		
		@Override
		public float[][] genVectors(int number, int dimension) {
			if (myRatio.length < dimension) {
				throw new RuntimeException("Not enough ratios");
			}
			
			float total = MU.sumToIndex(myRatio, dimension-1);
			int[] numNeeded = new int[dimension];
			for (int i = 0; i < numNeeded.length; i++) {
				numNeeded[i] = Math.round((float) number * myRatio[i] / total);
			}			
			int[] numGenerated = new int[dimension];
			boolean[] done = new boolean[dimension];
			
			float[][] result = new float[number][];
			int index = 0;
			while (!all(done)) {
				float[] vector = super.genVectors(1, dimension)[0];
				int biggestDim = biggestDimension(vector);
				if (!done[biggestDim]) {
					numGenerated[biggestDim]++;
					result[index] = vector;
					index++;
					if (numGenerated[biggestDim] == numNeeded[biggestDim]) done[biggestDim] = true;
				}
			}
			
			return result;
		}
		
		private static boolean all(boolean[] yes) {
			boolean result = true;
			for (boolean b : yes) {
				if (!b) result = false;
			}
			return result;
		}
		
		private static int biggestDimension(float[] vector) {
			float biggest = 0;
			int result = 0;
			
			for (int i = 0; i < vector.length; i++) {
				float size = Math.abs(vector[i]); 
				if (size > biggest) {
					biggest = size;
					result = i;
				}
			}
			
			return result;			
		}
		
	}
	
}
