<?php
require_once('vendor/autoload.php');
require 'src/Utility.php';


class UtilityTest extends PHPUnit\Framework\TestCase {

    protected $nn,$util,$path;
    protected $labels,$weight_i_h,$weight_h_o;
    protected $input_nodes,$hidden_nodes,$output_nodes,$lr;

    protected function setUp() {
        $this->input_nodes = 2;
        $this->hidden_nodes = 3;
        $this->output_nodes = 1;        
        $this->lr = 0.01;
        $this->nn = new NeuralNetwork\NeuralNetwork(
        $this->input_nodes,$this->hidden_nodes,$this->output_nodes,$this->lr
        );

        $this->util = new Utility\Utility();
        $this->labels = ["0","1"];
        $this->nn->setLabels($this->labels);      
        $this->weight_i_h = [[1,2,3],[4,5,6]];
        $this->weight_h_o = [[7],[8],[9]];
        $this->nn->setWeightIH($this->weight_i_h);
        $this->nn->setWeightHO($this->weight_h_o);

        $this->path = "models/test.dat";        
    }

    public function test_saveModel() {

     $this->assertTrue($this->util->saveModel($this->nn,$this->path));
     $this->assertTrue(is_file($this->path));

  }

  public function test_loadModel() {

    $model = $this->util->loadModel($this->path);

    $this->assertEquals($this->input_nodes,$model->getInputNodes());
    $this->assertEquals($this->hidden_nodes,$model->getHiddenNodes());
    $this->assertEquals($this->output_nodes,$model->getOutputNodes());
    $this->assertEquals($this->lr,$model->getLr());
    $this->assertEquals($this->labels,$model->getLabels());    
    $this->assertEquals($this->weight_i_h,$model->getWeightIH());    
    $this->assertEquals($this->weight_h_o,$model->getWeightHO());    

}

}
