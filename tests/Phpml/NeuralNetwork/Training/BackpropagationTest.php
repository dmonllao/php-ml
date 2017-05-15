<?php

declare(strict_types=1);

namespace tests\Phpml\NeuralNetwork\Training;

use Phpml\NeuralNetwork\Network\MultilayerPerceptron;
use Phpml\NeuralNetwork\Training\Backpropagation;
use PHPUnit\Framework\TestCase;

class BackpropagationTest extends TestCase
{
    public function testBackpropagationForXORLearning()
    {
        $network = new MultilayerPerceptron([2, 2, 1]);
        $training = new Backpropagation($network);

        $training->train(
            [[1, 0], [0, 1], [1, 1], [0, 0]],
            [[1], [1], [0], [0]],
            $desiredError = 0.3,
            40000
        );

        $this->assertEquals(0, $network->setInput([1, 1])->getOutput()[0], '', $desiredError);
        $this->assertEquals(0, $network->setInput([0, 0])->getOutput()[0], '', $desiredError);
        $this->assertEquals(1, $network->setInput([1, 0])->getOutput()[0], '', $desiredError);
        $this->assertEquals(1, $network->setInput([0, 1])->getOutput()[0], '', $desiredError);
    }

    public function testBackpropagationForXORLearningNoDesiredError()
    {
        // Without desiredError
        $network = new MultilayerPerceptron([2, 2, 1]);
        $training = new Backpropagation($network);

        $training->train(
            [[1, 0], [0, 1], [1, 1], [0, 0]],
            [[1], [1], [0], [0]],
            $desiredError = 0,
            40000
        );

        $output = $network->setInput([1, 1])->getOutput()[0];
        $this->assertLessThan(0.3, $output, '', $desiredError);
        $output = $network->setInput([0, 0])->getOutput()[0];
        $this->assertLessThan(0.3, $output, '', $desiredError);
        $output = $network->setInput([1, 0])->getOutput()[0];
        $this->assertGreaterThan(0.7, $output, '', $desiredError);
        $output = $network->setInput([0, 1])->getOutput()[0];
        $this->assertGreaterThan(0.7, $output, '', $desiredError);
    }

    public function testBackpropagationForXORLearningMultiLayer()
    {
        $network = new MultilayerPerceptron([2, 2, 2, 1]);
        $training = new Backpropagation($network);

        $training->train(
            [[1, 0], [0, 1], [1, 1], [0, 0]],
            [[1], [1], [0], [0]],
            $desiredError = 0.3,
            40000
        );

        $this->assertEquals(0, $network->setInput([1, 1])->getOutput()[0], '', $desiredError);
        $this->assertEquals(0, $network->setInput([0, 0])->getOutput()[0], '', $desiredError);
        $this->assertEquals(1, $network->setInput([1, 0])->getOutput()[0], '', $desiredError);
        $this->assertEquals(1, $network->setInput([0, 1])->getOutput()[0], '', $desiredError);
    }
}
