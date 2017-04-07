<?php

declare(strict_types=1);

namespace Phpml;

interface PartialTrainer
{

    /**
     * @param array $samples
     * @param array $targets
     * @param array $labels
     */
    public function partialTrain(array $samples, array $targets, array $labels = array());
}
