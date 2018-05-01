/**
 * 
 */
package com.bptripp.diff;

import ca.nengo.dynamics.Integrator;
import ca.nengo.dynamics.impl.AbstractDynamicalSystem;
import ca.nengo.dynamics.impl.EulerIntegrator;
import ca.nengo.math.Function;
import ca.nengo.math.impl.PostfixFunction;
import ca.nengo.model.Units;
import ca.nengo.plot.Plotter;
import ca.nengo.util.MU;
import ca.nengo.util.TimeSeries;
import ca.nengo.util.impl.TimeSeries1DImpl;

/**
 * Dynamics of short-term synaptic depression. 
 * 
 * @author Bryan Tripp
 */
public class SynapticDepressionDynamics extends AbstractDynamicalSystem {

	private static final long serialVersionUID = 1L;
	
	private int myMaxPoolSize;
	private float myTau;
	private float myProportionReleased;

	/**
	 * @param maxPoolSize Size of readily-releasable pool of synaptic vesicles 
	 * @param tau Time constant with which pool is replenished 
	 * @param proportionReleased Fraction of pool released with each spike 
	 */
	public SynapticDepressionDynamics(int maxPoolSize, float tau, float proportionReleased) {
		super(new float[1]);
		myMaxPoolSize = maxPoolSize;
		myTau = tau;
		myProportionReleased = proportionReleased;
	}

	/**
	 * @see ca.nengo.dynamics.impl.AbstractDynamicalSystem#f(float, float[])
	 */
	@Override
	public float[] f(float t, float[] u) {
		float S = getState()[0];
		return new float[]{(1-S) / myTau - myProportionReleased*S*u[0]};
	}

	/**
	 * @see ca.nengo.dynamics.impl.AbstractDynamicalSystem#g(float, float[])
	 */
	@Override
	public float[] g(float t, float[] u) {
		return new float[]{Math.round((float) myMaxPoolSize * getState()[0]) / (float) myMaxPoolSize};
	}

	/**
	 * @see ca.nengo.dynamics.impl.AbstractDynamicalSystem#getInputDimension()
	 */
	@Override
	public int getInputDimension() {
		return 1;
	}

	/**
	 * @see ca.nengo.dynamics.impl.AbstractDynamicalSystem#getOutputDimension()
	 */
	@Override
	public int getOutputDimension() {
		return 1;
	}
	
	/**
	 * @return Time constant with which pool is replenished 
	 */
	public float getTau() {
		return myTau;
	}
	
	/**
	 * @param tau Time constant with which pool is replenished 
	 */
	public void setTau(float tau) {
		myTau = tau;
	}

	/**
	 * @return Fraction of pool released with each spike 
	 */
	public float getProportionReleased() {
		return myProportionReleased;
	}

	/**
	 * @param proportion Fraction of pool released with each spike 
	 */
	public void setProportionReleased(float proportion) {
		myProportionReleased = proportion;
	}

	/** 
	 * Demonstration code -- empty args list OK. 
	 */
	public static void main(String[] args) {
		SynapticDepressionDynamics dynamics = new SynapticDepressionDynamics(100, .5f, .01f);
		Integrator integrator = new EulerIntegrator(.001f);
		float[] times = MU.makeVector(0, .001f, 5);
		Function f = new PostfixFunction("200 + 100*sin(2*pi*x0)", 1);
		float[] values = f.multiMap(MU.transpose(new float[][]{times}));
		TimeSeries input = new TimeSeries1DImpl(times, values, Units.UNK);
		TimeSeries output = integrator.integrate(dynamics, input);
		Plotter.plot(input, "input");
		Plotter.plot(output, "output");
		Plotter.plot(MU.prodElementwise(values, MU.transpose(output.getValues())[0]), "product");
	}
}
