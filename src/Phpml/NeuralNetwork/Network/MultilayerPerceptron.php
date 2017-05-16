<?php

declare(strict_types=1);

namespace Phpml\NeuralNetwork\Network;

use Phpml\Exception\InvalidArgumentException;
use Phpml\NeuralNetwork\ActivationFunction;
use Phpml\NeuralNetwork\Layer;
use Phpml\NeuralNetwork\Node\Bias;
use Phpml\NeuralNetwork\Node\Input;
use Phpml\NeuralNetwork\Node\Neuron;
use Phpml\NeuralNetwork\Node\Neuron\Synapse;
use Phpml\Helper\Predictable;

class MultilayerPerceptron extends LayeredNetwork
{
    use Predictable;

    /**
     * @var array
     */
    private $classes = [];

    /**
     * @param int                     $inputLayerFeatures
     * @param array                   $hiddenLayers
     * @param array                   $classes
     * @param ActivationFunction|null $activationFunction
     *
     * @throws InvalidArgumentException
     */
    public function __construct(int $inputLayerFeatures, array $hiddenLayers, array $classes, ActivationFunction $activationFunction = null)
    {
        if (count($hiddenLayers) < 1) {
            throw InvalidArgumentException::invalidLayersNumber();
        }

        $nClasses = count($classes);
        if ($nClasses < 2) {
            throw InvalidArgumentException::invalidClassesNumber();
        }
        $this->classes = $classes;

        $this->addInputLayer($inputLayerFeatures);
        $this->addNeuronLayers($hiddenLayers, $activationFunction);
        $this->addNeuronLayers([$nClasses], $activationFunction);

        $this->addBiasNodes();
        $this->generateSynapses();
    }

    /**
     * @param  mixed $target
     * @return int
     */
    public function getTargetClass($target): int
    {
        if (in_array($target, $this->classes) === false) {
            throw InvalidArgumentException::invalidTarget($target);
        }
        return array_search($target, $this->classes);
    }

    /**
     * @param array $sample
     *
     * @return mixed
     */
    public function predictSample(array $sample)
    {
        $output = $this->setInput($sample)->getOutput();

        $predictedClass = null;
        $max = 0;
        foreach ($output as $class => $value) {
            if ($value > $max) {
                $predictedClass = $class;
                $max = $value;
            }
        }
        return $this->classes[$predictedClass];
    }

    /**
     * @param int $nodes
     */
    private function addInputLayer(int $nodes)
    {
        $this->addLayer(new Layer($nodes, Input::class));
    }

    /**
     * @param array                   $layers
     * @param ActivationFunction|null $activationFunction
     */
    private function addNeuronLayers(array $layers, ActivationFunction $activationFunction = null)
    {
        foreach ($layers as $neurons) {
            $this->addLayer(new Layer($neurons, Neuron::class, $activationFunction));
        }
    }

    private function generateSynapses()
    {
        $layersNumber = count($this->layers) - 1;
        for ($i = 0; $i < $layersNumber; ++$i) {
            $currentLayer = $this->layers[$i];
            $nextLayer = $this->layers[$i + 1];
            $this->generateLayerSynapses($nextLayer, $currentLayer);
        }
    }

    private function addBiasNodes()
    {
        $biasLayers = count($this->layers) - 1;
        for ($i = 0; $i < $biasLayers; ++$i) {
            $this->layers[$i]->addNode(new Bias());
        }
    }

    /**
     * @param Layer $nextLayer
     * @param Layer $currentLayer
     */
    private function generateLayerSynapses(Layer $nextLayer, Layer $currentLayer)
    {
        foreach ($nextLayer->getNodes() as $nextNeuron) {
            if ($nextNeuron instanceof Neuron) {
                $this->generateNeuronSynapses($currentLayer, $nextNeuron);
            }
        }
    }

    /**
     * @param Layer  $currentLayer
     * @param Neuron $nextNeuron
     */
    private function generateNeuronSynapses(Layer $currentLayer, Neuron $nextNeuron)
    {
        foreach ($currentLayer->getNodes() as $currentNeuron) {
            $nextNeuron->addSynapse(new Synapse($currentNeuron));
        }
    }
}
