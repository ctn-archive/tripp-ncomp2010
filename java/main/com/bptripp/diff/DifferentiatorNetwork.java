/*
 * Created on 31-Mar-08
 */
package com.bptripp.diff;

import java.io.File;
import java.io.IOException;

import ca.nengo.dynamics.DynamicalSystem;
import ca.nengo.dynamics.Integrator;
import ca.nengo.dynamics.impl.EulerIntegrator;
import ca.nengo.dynamics.impl.LTISystem;
import ca.nengo.io.MatlabExporter;
import ca.nengo.math.Function;
import ca.nengo.math.impl.GaussianPDF;
import ca.nengo.math.impl.IndicatorPDF;
import ca.nengo.math.impl.SineFunction;
import ca.nengo.model.Ensemble;
import ca.nengo.model.Node;
import ca.nengo.model.Noise;
import ca.nengo.model.Projection;
import ca.nengo.model.SimulationException;
import ca.nengo.model.StructuralException;
import ca.nengo.model.Units;
import ca.nengo.model.impl.FunctionInput;
import ca.nengo.model.impl.NetworkImpl;
import ca.nengo.model.impl.NoiseFactory;
import ca.nengo.model.nef.NEFEnsemble;
import ca.nengo.model.nef.NEFEnsembleFactory;
import ca.nengo.model.nef.impl.NEFEnsembleFactoryImpl;
import ca.nengo.model.neuron.impl.LIFNeuronFactory;
import ca.nengo.util.MU;
import ca.nengo.util.Probe;
import ca.nengo.util.TimeSeries;

/**
 * A Network that performs temporal differentiation. This interface provides 
 * accessors to some common properties of differentiator networks. 
 * 
 * @author Bryan Tripp
 */
public abstract class DifferentiatorNetwork extends NetworkImpl {
	
	private static final long serialVersionUID = 1L;

	public static final float TAU_IO = .005f;
	public static float TAU_INTERNEURONS = .001f;
	
	private FunctionInput myInput;
	private NEFEnsemble myInputEnsemble;
	private NEFEnsemble myOutputEnsemble;
	private int myNInput = 2000; 
	private int myNOutput = 1000;
	protected transient NEFEnsembleFactory myEnsembleFactory; 	
	private Probe myInputProbe;
	private Probe myOutputProbe;
	private Probe myInputEnsembleProbe;
	
	public DifferentiatorNetwork() throws StructuralException {
		myEnsembleFactory = new NEFEnsembleFactoryImpl();
		
		myInput = new FunctionInput("external", new Function[]{new SineFunction((float) Math.PI, 1f/ (float) Math.PI)}, Units.UNK);
		addNode(myInput);
		
		myInputEnsemble = myEnsembleFactory.make("input", myNInput, 1, "diff_input_"+myNInput, false);
		addNode(myInputEnsemble);

		//the output ensemble contains near-linear neurons 
		NEFEnsembleFactoryImpl of = new NEFEnsembleFactoryImpl();
		of.setNodeFactory(new LIFNeuronFactory(.02f, .0001f, new IndicatorPDF(200, 400), new IndicatorPDF(-.9f, .9f)));
		myOutputEnsemble = of.make("output", myNOutput, 1, "diff_output_"+myNOutput, false);		
		addNode(myOutputEnsemble);
		
		try {
			myInputProbe = getSimulator().addProbe(myInput.getName(), FunctionInput.STATE_NAME, true);
			myInputEnsembleProbe = getSimulator().addProbe(myInputEnsemble.getName(), NEFEnsemble.X, true);
			myOutputProbe = getSimulator().addProbe(myOutputEnsemble.getName(), NEFEnsemble.X, true);
		} catch (SimulationException e) {
			throw new StructuralException(e);
		}
	}
	
	@Override
	public void reset(boolean randomize) {
		super.reset(randomize);
		for (Probe p : getSimulator().getProbes()) {
			p.reset();
		}
	}

	/**
	 * @return the abstract external input
	 */
	protected FunctionInput getInput() {
		return myInput;
	}

	/**
	 * @return the ensemble that encodes the input value
	 */
	protected NEFEnsemble getInputEnsemble() {
		return myInputEnsemble;
	}
	
	/**
	 * @return the ensemble that encodes the output value
	 */
	protected NEFEnsemble getOutputEnsemble() {
		return myOutputEnsemble;
	}
	
	/**
	 * @return Abstract input value from last run
	 */
	public TimeSeries getInputData() {
		return myInputProbe.getData();
	}
	
	/**
	 * @return decoded input representation from last run
	 */
	public TimeSeries getInputEnsembleData() {
		return myInputEnsembleProbe.getData();
	}
	
	/**
	 * @return Decoded output representation from last run
	 */
	public TimeSeries getOutputData() {
		return myOutputProbe.getData();
	}
	
	/**
	 * @param input External input to the network (a function of time)
	 * @throws StructuralException
	 */
	public void setInputFunction(Function input) throws StructuralException {
		myInput.setFunctions(new Function[]{input});
	}
	
	/**
	 * @param tau PSC time constant for differentiator ensembles  
	 */
	public abstract void setTau(float tau);
	
	/**
	 * Sets parameters that determine the amplitudes of abstract noise models (intended for abstract runs in which neurons aren't used). 
	 *  
	 * @param nInput Nominal number of input neurons 
	 * @param nDiff Nominal number of output neurons
	 */
	public abstract void setNoise(int nInput, int nDiff);

	/**
	 * Sets parameters that determine the amplitudes of abstract distortion models (intended for abstract runs in which neurons aren't used).
	 * 
	 * @param nInput Nominal number of input neurons 
	 * @param nDiff Nominal number of output neurons
	 */
	public abstract void setDistortion(int nInput, int nDiff);

	/**
	 * Removes any existing abstract noise and distortion models. 
	 */
	public abstract void clearErrors();

	/**
	 * Converts idealized projections with mixed-sign synaptic weights to the more realistic form described by 
	 * Parisien et al., 2008. 
	 *  
	 * @param propInhibitory Number of inhibitory neurons in each projection, as a fraction of the number of target neurons. 
	 *   
	 * @throws StructuralException
	 */
	public abstract void enableParisien(float propInhibitory) throws StructuralException;
	
	/**
	 * Removes Parisien-related additions, returning projections to abstract mixed-sign synaptic weights. 
	 */
	public abstract void disableParisien();
	
	/**
	 * Converts a single projection to Parisien form. 
	 * 
	 * @param p The projection to convert. 
	 * @param n Number of inhibitory neurons. 
	 * 
	 * @throws StructuralException
	 */
	protected static void enableParisien(Projection p, int n) throws StructuralException {
		p.addBias(n, TAU_INTERNEURONS, p.getTermination().getTau(), true, true);
	}

	/**
	 * Converts a single projection to Parisien form. 
	 * 
	 * @param p The projection to convert. 
	 * @param n Number of inhibitory neurons. 
	 * @param optimize if true, apply some performance optimizations not discussed by Parisien et al.  
	 * @throws StructuralException
	 */
	protected static void enableParisien(Projection p, int n, boolean optimize) throws StructuralException {
		p.addBias(n, TAU_INTERNEURONS, p.getTermination().getTau(), true, optimize);
	}
	
	/**
	 * @param destination File to which to write matlab export
	 * @throws IOException
	 */
	public void exportAll(File destination) throws IOException {
		MatlabExporter exporter = new MatlabExporter();

		Probe[] probes = getSimulator().getProbes();
		for (int i = 0; i < probes.length; i++) {
			TimeSeries data = probes[i].getData();
			exporter.add(data.getName().replace(':', '_'), data);
		}
		
		Node[] nodes = getNodes();
		for (int i = 0; i < nodes.length; i++) {
			if (nodes[i] instanceof Ensemble && ((Ensemble) nodes[i]).isCollectingSpikes()) {
				exporter.add(nodes[i].getName(), ((Ensemble) nodes[i]).getSpikePattern());
			}
		}
		
		exporter.write(destination);
	}
	
	/**
	 * @param n Nominal number of presynaptic neurons in a projection 
	 * @return A model of noise arising within the projection
	 */
	protected static Noise makeNoise(int n) {
		return makeNoise(1f/n, 1000, true);
	}
	
	/**
	 * @param n Nominal number of presynaptic neurons in a projection 
	 * @return A model of distortion error arising within the projection
	 */
	protected static Noise makeDistortion(int n) {
		return makeNoise(1f/(n*n), 400, true);
	}
	
	/**
	 * @param variance Noise power 
	 * @param frequency Sampling frequency of noise process
	 * @param filter Filter applied to noise to control bandwidth
	 * @return A Noise model conforming to the given specs
	 */
	protected static Noise makeNoise(float variance, float frequency, boolean filter) {
		DynamicalSystem noiseFilter = null;
		if (filter) {
			noiseFilter = getNoiseFilter(frequency / 2f);
		}
		
		Integrator integrator = new EulerIntegrator(.0005f);
		return NoiseFactory.makeRandomNoise(frequency, new GaussianPDF(0, variance), noiseFilter, integrator);
	}
	
	private static DynamicalSystem getNoiseFilter(float frequency) {
		float w = 2 * (float) Math.PI * frequency;
		float[][] A = new float[][]{new float[]{0, 1}, new float[]{-w*w, -(float)Math.sqrt(2)*w}};
		float[][] B = new float[][]{new float[]{0}, new float[]{w*w}};
		float[][] C = new float[][]{new float[]{1, 0}};
		DynamicalSystem result = new LTISystem(A, B, C, MU.zero(1, 1), new float[]{0, 0}, new Units[]{Units.UNK});
		return result;
	}
	
}
