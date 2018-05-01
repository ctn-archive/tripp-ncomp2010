/*
 * Created on 10-May-07
 */
package com.bptripp.diff;

import ca.nengo.math.Function;
import ca.nengo.math.PDF;
import ca.nengo.math.impl.AbstractFunction;
import ca.nengo.math.impl.ConstantFunction;
import ca.nengo.math.impl.GaussianPDF;
import ca.nengo.math.impl.LinearCurveFitter;
import ca.nengo.math.impl.PiecewiseConstantFunction;
import ca.nengo.model.Network;
import ca.nengo.model.SimulationException;
import ca.nengo.model.StructuralException;
import ca.nengo.model.Termination;
import ca.nengo.model.Units;
import ca.nengo.model.impl.FunctionInput;
import ca.nengo.model.impl.NetworkImpl;
import ca.nengo.model.nef.NEFEnsemble;
import ca.nengo.util.MU;
import ca.nengo.util.Probe;
import ca.nengo.util.TimeSeries;

/**
 * Utility methods. 
 * 
 * @author Bryan Tripp
 */
public class Util {

	/**
	 * Finds steady-state output of an NEFEnsemble over a range of inputs and returns a Function 
	 * that approximates its negative. NOTE: there can't be any terminations on the ensemble when this
	 * method is called (add them later). 
	 * 
	 * @param ensemble An ensemble for which bias is to be found
	 * @param origin Name of DecodedOrigin of interest on given ensemble
	 * @return A Function that can be added to the output of the given Origin to cancel it at steady state 
	 * @throws StructuralException
	 * @throws SimulationException
	 */
	public static Function getBiasCompensation(NEFEnsemble ensemble, String origin, float transientTime) throws StructuralException, SimulationException {		
		float simulationTime = transientTime * 2 + .2f;
		float endTime = simulationTime * .95f;
		
		Network network = new NetworkImpl();
		Termination t = ensemble.addDecodedTermination("laskdjhcuwyge19238479DLSKFASDKFJH", MU.I(1), .001f, false);
		network.addNode(ensemble);		
		FunctionInput input = new FunctionInput("testinput", new Function[]{new ConstantFunction(1, 0f)}, Units.UNK);
		network.addNode(input);
		network.addProjection(input.getOrigin(FunctionInput.ORIGIN_NAME), t);
		Probe pOut = network.getSimulator().addProbe(ensemble.getName(), origin, true);
		
		float[] x = MU.makeVector(-1f, .1f, 1f);
		float[] compensation = new float[x.length];
		for (int i = 0; i < x.length; i++) {
			Function f = new PiecewiseConstantFunction(new float[]{.1f}, new float[]{0, x[i]});
			input.setFunctions(new Function[]{f});
			network.reset(false);
			network.run(0, simulationTime);
			TimeSeries output = pOut.getData();
			
			float sum = 0, count = 0;
			for (int j = 0; j < output.getTimes().length; j++) {
				if (output.getTimes()[j] >= transientTime && output.getTimes()[j] <= endTime) {
					count++;
					sum += output.getValues()[j][0];
				}
			}
			compensation[i] = -sum/count;
			
			pOut.reset();
			network.reset(false);
		}
		ensemble.removeDecodedTermination(t.getName());

		return new LinearCurveFitter().fit(x, compensation);
	}

	public static Function RAMP = new AbstractFunction(1) {
		private static final long serialVersionUID = 1L;
		private PDF myNoisePDF = new GaussianPDF(0, .00025f);
		public float map(float[] from) {
			return myNoisePDF.sample()[0] + .75f * Math.max(0, ((from[0] < 1.5) ? from[0] - .5f : 2.5f - from[0])); 
		}
	};
}
