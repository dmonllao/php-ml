<?php

declare(strict_types=1);

namespace Phpml\NeuralNetwork\Training;

use Phpml\NeuralNetwork\Network;
use Phpml\NeuralNetwork\Node\Neuron;
use Phpml\NeuralNetwork\Training;
use Phpml\NeuralNetwork\Training\Backpropagation\Sigma;

class Backpropagation implements Training
{
    /**
     * @var Network
     */
    private $network;

    /**
     * @var int
     */
    private $theta;

    /**
     * @var int
     */
    private $maxIterations;

    /**
     * @var array
     */
    private $sigmas;

    /**
     * @var array
     */
    private $prevSigmas;

    /**
     * @param Network $network
     * @param int     $theta
     */
    public function __construct(Network $network, int $theta = 1, int $maxIterations = 10000)
    {
        $this->network = $network;
        $this->theta = $theta;
        $this->maxIterations = $maxIterations;
    }

    /**
     * @param array $samples
     * @param array $targets
     */
    public function train(array $samples, array $targets)
    {
        for ($i = 0; $i < $this->maxIterations; ++$i) {
            $this->trainSamples($samples, $targets);
        }
    }

    /**
     * @param array $samples
     * @param array $targets
     */
    private function trainSamples(array $samples, array $targets)
    {
        foreach ($targets as $key => $target) {
            $this->trainSample($samples[$key], $target);
        }
    }

    /**
     * @param array $sample
     * @param mixed $target
     */
    private function trainSample(array $sample, $target)
    {

        // Feed-forward.
        $this->network->setInput($sample)->getOutput();

        $layers = $this->network->getLayers();
        $layersNumber = count($layers);

        $targetClass = $this->network->getTargetClass($target);

        // Backpropagation.
        for ($i = $layersNumber; $i > 1; --$i) {
            $this->sigmas = [];
            foreach ($layers[$i - 1]->getNodes() as $key => $neuron) {

                if ($neuron instanceof Neuron) {
                    $sigma = $this->getSigma($neuron, $targetClass, $key, $i == $layersNumber);
                    foreach ($neuron->getSynapses() as $synapse) {
                        $synapse->changeWeight($this->theta * $sigma * $synapse->getNode()->getOutput());
                    }
                }
            }
            $this->prevSigmas = $this->sigmas;
        }
    }

    /**
     * @param Neuron $neuron
     * @param int    $targetClass
     * @param int    $key
     * @param bool   $lastLayer
     *
     * @return float
     */
    private function getSigma(Neuron $neuron, int $targetClass, int $key, bool $lastLayer): float
    {
        $neuronOutput = $neuron->getOutput();
        $sigma = $neuronOutput * (1 - $neuronOutput);

        if ($lastLayer) {
            if ($targetClass == $key) {
                $value = 1;
            } else {
                $value = 0;
            }
            $sigma *= ($value - $neuronOutput);
        } else {
            $sigma *= $this->getPrevSigma($neuron);
        }

        $this->sigmas[] = new Sigma($neuron, $sigma);

        return $sigma;
    }

    /**
     * @param Neuron $neuron
     *
     * @return float
     */
    private function getPrevSigma(Neuron $neuron): float
    {
        $sigma = 0.0;

        foreach ($this->prevSigmas as $neuronSigma) {
            $sigma += $neuronSigma->getSigmaForNeuron($neuron);
        }

        return $sigma;
    }
}
