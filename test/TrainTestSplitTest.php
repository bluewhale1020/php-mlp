<?php

use CrossValidation\TrainTestSplit;
require_once('vendor/autoload.php');
require 'src/TrainTestSplit.php';
require 'src/DatasetManager.php';

class TrainTestSplitTest extends PHPUnit\Framework\TestCase {

    protected $split;
    protected $dataset;

    protected function setUp() {
        $this->split = new TrainTestSplit();
        $path = "./dataset/test2.csv";
        $this->dataset = new Dataset\DatasetManager($path,true);

    }

    public function test_run() {

        $result = $this->split->run($this->dataset);
        
        print_r($result['train']);
        print_r($result['test']);
        // $this->assertEquals($this->weight_h_o,$model->getWeightHO()); 
        // $features =[[0,1],[1,0],[1,1],[0,0]];
        // $target = [1,1,0,0];
      
        // $this->dataset->setFeatures($features);
        // $this->dataset->setTargets($target);        
        
        // $result = $this->split->run($this->dataset);
        
        // print_r($result['train']);
        // print_r($result['test']);    
    }

}