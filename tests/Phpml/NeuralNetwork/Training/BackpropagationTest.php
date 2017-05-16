<?php

declare(strict_types=1);

namespace tests\Phpml\NeuralNetwork\Training;

use Phpml\NeuralNetwork\Network\MultilayerPerceptron;
use Phpml\NeuralNetwork\Training\Backpropagation;
use PHPUnit\Framework\TestCase;

class BackpropagationTest extends TestCase
{
    public function testBackpropagationLearning()
    {
        // Single layer 2 classes.
        $network = new MultilayerPerceptron(2, [2], ['a', 'b']);
        $training = new Backpropagation($network, 1, 1000);

        $training->train(
            [[1, 0], [0, 1], [1, 1], [0, 0]],
            ['a', 'b', 'a', 'b']
        );

        $this->assertEquals('a', $network->predict([1, 0]));
        $this->assertEquals('b', $network->predict([0, 1]));
        $this->assertEquals('a', $network->predict([1, 1]));
        $this->assertEquals('b', $network->predict([0, 0]));
    }

    public function testBackpropagationLearningMultilayer()
    {
        // Multi-layer 2 classes.
        $network = new MultilayerPerceptron(5, [3, 2], ['a', 'b']);
        $training = new Backpropagation($network);

        $training->train(
            [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
            ['a', 'b', 'a', 'b']
        );

        $this->assertEquals('a', $network->predict([1, 0, 0, 0, 0]));
        $this->assertEquals('b', $network->predict([0, 1, 1, 0, 0]));
        $this->assertEquals('a', $network->predict([1, 1, 1, 1, 1]));
        $this->assertEquals('b', $network->predict([0, 0, 0, 0, 0]));
    }

    public function testBackpropagationLearningMulticlass()
    {
        // Multi-layer more than 2 classes.
        $network = new MultilayerPerceptron(5, [3, 2], ['a', 'b', 4]);
        $training = new Backpropagation($network);

        $training->train(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
            ['a', 'b', 'a', 'a', 4]
        );

        $this->assertEquals('a', $network->predict([1, 0, 0, 0, 0]));
        $this->assertEquals('b', $network->predict([0, 1, 0, 0, 0]));
        $this->assertEquals('a', $network->predict([0, 0, 1, 1, 0]));
        $this->assertEquals('a', $network->predict([1, 1, 1, 1, 1]));
        $this->assertEquals(4, $network->predict([0, 0, 0, 0, 0]));
    }
}
