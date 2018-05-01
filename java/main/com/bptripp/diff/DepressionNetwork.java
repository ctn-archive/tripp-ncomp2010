/*
 * Created on 11-Apr-08
 */
package com.bptripp.diff;

import ca.nengo.dynamics.DynamicalSystem;
import ca.nengo.math.Function;
import ca.nengo.math.impl.IndicatorPDF;
import ca.nengo.model.Node;
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
import ca.nengo.model.nef.impl.BiasOrigin;
import ca.nengo.model.nef.impl.DecodedOrigin;
import ca.nengo.model.nef.impl.DecodedTermination;
import ca.nengo.model.nef.impl.NEFEnsembleFactoryImpl;
import ca.nengo.model.neuron.Neuron;
import ca.nengo.model.neuron.impl.LIFNeuronFactory;
import ca.nengo.model.neuron.impl.SpikingNeuron;
import ca.nengo.util.MU;
import ca.nengo.util.Probe;
import ca.nengo.util.TimeSeries;

/**
 * A DifferentiatorNetwork in which differentiation is achieved through short-term synaptic depression. 
 *   
 * @author Bryan Tripp
 */
public class DepressionNetwork extends DifferentiatorNetwork {

	private static final long serialVersionUID = 1L;
	
	private static final String DEPRESSING = "depressing";
	private static final String COMPENSATING = "compensating";

	private NEFEnsemble myDepressingEnsemble;
	private Probe myInputProbe;
	private Projection myDepressingProjection;
	private Projection myCompensatingProjection;

	/**
	 * @param n Number of neurons with depressing synapses (presynaptic depression mechanisms). 
	 *   
	 * @throws StructuralException
	 */
	public DepressionNetwork(int n) throws StructuralException {
		setName("depression");
		
		removeNode(super.getInputEnsemble().getName());
		for (Probe probe : getSimulator().getProbes()) {
			if (probe.getTarget().equals(super.getInputEnsemble())) {
				try {
					getSimulator().removeProbe(probe);
				} catch (SimulationException e) {
					throw new RuntimeException(e);
				}
			}
		}

		String name = super.getInputEnsemble().getName();
		
		myDepressingEnsemble = getLinearFactory().make(name, n, 1, "depression_input_"+n, true);
		myDepressingEnsemble.addDecodedTermination("input", MU.I(1), TAU_IO, false);
		addNode(myDepressingEnsemble);
		try {
			myInputProbe = getSimulator().addProbe(myDepressingEnsemble.getName(), NEFEnsemble.X, true);
		} catch (SimulationException e) {
			throw new RuntimeException(e);
		}
		addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), getInputEnsemble().getTermination("input"));
		
		int maxPoolSize = 100;
		float tauRecovery = 0.5f;
		float proportionReleased = .01f;
		DynamicalSystem depressionDynamics = new SynapticDepressionDynamics(maxPoolSize, tauRecovery, proportionReleased);
		((DecodedOrigin) myDepressingEnsemble.getOrigin(NEFEnsemble.X)).setSTPDynamics(depressionDynamics);

		NEFEnsemble output = getOutputEnsemble();
		float scale = 1f;
		output.addDecodedTermination(DEPRESSING, new float[][]{new float[]{scale}}, TAU_IO, false);
		output.addDecodedTermination(COMPENSATING, new float[][]{new float[]{scale}}, TAU_IO, false);
		
		setCompensation(.1f);		
		addProjections();
	}
	
	private static NEFEnsembleFactory getLinearFactory() {
		NEFEnsembleFactory result = new NEFEnsembleFactoryImpl();
		result.setNodeFactory(new LIFNeuronFactory(.02f, .0005f, new IndicatorPDF(200, 400), new IndicatorPDF(-2.5f, -1.5f)));
		return result;
	}
	
	private void removeProjections() throws StructuralException {
		removeProjection(getInputEnsemble().getTermination("input"));
		removeProjection(getOutputEnsemble().getTermination(DEPRESSING));					
		removeProjection(getOutputEnsemble().getTermination(COMPENSATING));					
	}
	
	private void addProjections() throws StructuralException {
		addProjection(getInput().getOrigin(FunctionInput.ORIGIN_NAME), getInputEnsemble().getTermination("input"));
		myDepressingProjection = addProjection(myDepressingEnsemble.getOrigin(NEFEnsemble.X), getOutputEnsemble().getTermination(DEPRESSING));
		myCompensatingProjection = addProjection(myDepressingEnsemble.getOrigin(COMPENSATING), getOutputEnsemble().getTermination(COMPENSATING));			
	}
	
	@Override
	public void disableParisien() {
		myCompensatingProjection.removeBias();
		myDepressingProjection.removeBias();
	}

	@Override
	public void enableParisien(float propInhibitory) throws StructuralException {
		int n = Math.round(propInhibitory * (float) getOutputEnsemble().getNodes().length);
		enableParisien(myCompensatingProjection, n);
		
		enableParisien(myDepressingProjection, n, false);
		DecodedOrigin o = ((DecodedOrigin) myDepressingEnsemble.getOrigin(NEFEnsemble.X));
		BiasOrigin bo = ((BiasOrigin) myDepressingEnsemble.getOrigin("output:depressing")); 
		bo.setSTPDynamics(o.getSTPDynamics());
		for (int i = 0; i < bo.getDecoders().length; i++) {
			((SynapticDepressionDynamics) bo.getSTPDynamics(i)).setTau(
					((SynapticDepressionDynamics) o.getSTPDynamics(i)).getTau());
			((SynapticDepressionDynamics) bo.getSTPDynamics(i)).setProportionReleased(
					((SynapticDepressionDynamics) o.getSTPDynamics(i)).getProportionReleased());
		}
	}

	@Override
	protected NEFEnsemble getInputEnsemble() {
		return myDepressingEnsemble;
	}

	@Override
	public TimeSeries getInputEnsembleData() {
		return myInputProbe.getData();
	}

	@Override
	public void setTau(float tau) {
		Node[] neurons = myDepressingEnsemble.getNodes();
		try {
			DecodedOrigin o = (DecodedOrigin) myDepressingEnsemble.getOrigin(NEFEnsemble.X);
			for (int i = 0; i < neurons.length; i++) {
				SpikingNeuron neuron = (SpikingNeuron) neurons[i];
				float r0 = getNominalRate(neuron);
				
				//choose F so at equilibrium S = 1/2 at r0 (this determines tauS)
				float F = 1 / (2*tau*r0);
				float tauS = 2*tau;
				SynapticDepressionDynamics d = (SynapticDepressionDynamics) o.getSTPDynamics(i); //note the index i
				d.setTau(tauS);
				d.setProportionReleased(F);
				System.out.println("tauS: " + tauS + " F:" + F);
			}		
			
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
		
		try {
			removeProjections();
			setCompensation(tau);
			addProjections();

			float[][] scale = new float[][]{new float[]{4.4f / tau}};
			((DecodedTermination) getOutputEnsemble().getTermination(DEPRESSING)).setTransform(scale);
			((DecodedTermination) getOutputEnsemble().getTermination(COMPENSATING)).setTransform(scale);
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}		
	}
	
	private void setCompensation(float tau) throws StructuralException {
		try {
			myDepressingEnsemble.removeDecodedTermination("input"); //have to remove this temporarily
			Function f = Util.getBiasCompensation(myDepressingEnsemble, NEFEnsemble.X, tau*8);
			myDepressingEnsemble.addDecodedTermination("input", MU.I(1), TAU_IO, false);
			myDepressingEnsemble.addDecodedOrigin(COMPENSATING, new Function[]{f}, Neuron.AXON);
		} catch (SimulationException e) {
			throw new StructuralException(e);
		}
	}	
	
	private static float getNominalRate(SpikingNeuron neuron) {
		SimulationMode mode = neuron.getMode();
		float rate = 0;
		
		try {
			neuron.setMode(SimulationMode.CONSTANT_RATE);
			neuron.setRadialInput(0);
			neuron.run(0, 0);
			rate = ((RealOutput) neuron.getOrigin(Neuron.AXON).getValues()).getValues()[0];
			neuron.setMode(mode);
		} catch (SimulationException e) {
			throw new RuntimeException(e);
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
		
		return rate;
	}

	@Override
	public void clearErrors() {
		try {
			((Noisy) myDepressingEnsemble.getOrigin(NEFEnsemble.X)).setNoise(new NoiseFactory.NoiseImplNull());
			((Noisy) myDepressingEnsemble.getOrigin(COMPENSATING)).setNoise(new NoiseFactory.NoiseImplNull());
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void setDistortion(int nInput, int nDiff) {
		try {
			((Noisy) myDepressingEnsemble.getOrigin(NEFEnsemble.X)).setNoise(makeDistortion(nInput));
			((Noisy) myDepressingEnsemble.getOrigin(COMPENSATING)).setNoise(makeDistortion(nInput));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void setNoise(int nInput, int nDiff) {
		try {
			((Noisy) myDepressingEnsemble.getOrigin(NEFEnsemble.X)).setNoise(makeNoise(nInput));
			((Noisy) myDepressingEnsemble.getOrigin(COMPENSATING)).setNoise(makeNoise(nInput));
		} catch (StructuralException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static void main(String[] args) {
		try {
			DepressionNetwork n = new DepressionNetwork(500);
			n.setTau(.5f);
		} catch (StructuralException e) {
			e.printStackTrace();
		}
	}

}
