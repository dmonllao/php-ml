<?php

declare(strict_types=1);

namespace Phpml\Helper;

trait PartiallyTrainable
{

    /**
     * @param array $samples
     * @param array $targets
     */
    public function train(array $samples, array $targets)
    {

        // Clears previous labels and classifiers.
        $this->resetTrainer();

        return $this->partialTrain($samples, $targets);
    }

    /**
     * @param array $samples
     * @param array $targets
     * @param array $labels
     */
    abstract public function partialTrain(array $samples, array $targets, array $labels = array());

    /**
     * Clear the trainer before training again.
     */
    abstract protected function resetTrainer();
}
