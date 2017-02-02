<?php

declare(strict_types=1);

namespace tests;

use Phpml\ModelManager;
use Phpml\Regression\LeastSquares;

class ModelManagerTest extends \PHPUnit_Framework_TestCase
{

    public function testSaveAndRestore()
    {
        $filename = 'test-save-to-file-'.rand(100, 999).'-'.uniqid();
        $filepath = tempnam(sys_get_temp_dir(), $filename);

        $obj = new LeastSquares();
        $modelManager = new ModelManager();
        $modelManager->saveToFile($obj, $filepath);

        $restored = $modelManager->restoreFromFile($filepath);
        $this->assertEquals($obj, $restored);
    }

    /**
     * @expectedException \Phpml\Exception\FileException
     */
    public function testSaveToWrongFile()
    {
        $filepath = '/really-really-unlikely-to-exist-dir/save-unexisting-file';

        $obj = new LeastSquares();
        $modelManager = new ModelManager();
        $modelManager->saveToFile($obj, $filepath);
    }

    /**
     * @expectedException \Phpml\Exception\FileException
     */
    public function testRestoreWrongFile()
    {
        $filepath = '/really-really-unlikely-to-exist-dir/restore-unexisting-file';
        $modelManager = new ModelManager();
        $modelManager->restoreFromFile($filepath);
    }
}
