<?php
require_once('vendor/autoload.php');
require 'src/NeuralNetwork.php';

class NeuralNetworkTest extends PHPUnit\Framework\TestCase {

    protected $nn;
    protected $a,$b,$c,$vector;
 
    protected function setUp() {
        $this->nn = new NeuralNetwork\NeuralNetwork(2,3,1,0.1);
        $this->a = [[1,2,3], [4,5,6]];
        $this->b = [[7,8,9], [10,11,12]];
        $this->h = [[7,-8,9], [-10,11,-12]];        
        $this->c = [[13,14], [15,16], [17,18]];
        $this->vector = [3,9,2];
    }

    public function test_construct() {

      $weight_i_h = $this->nn->getWeightIH();
      //2 * 3
      print_r($weight_i_h);
      $weight_h_o = $this->nn->getWeightHO();
      //1 * 3
      //print_r($weight_h_o);

  }

    public function test_selectLabel() {
      $output = [2.1,0.9,0.1];
      $labels = ["e","f","g"];
      $this->nn->setLabels($labels);
      $expected = ["e"];
     $this->assertEquals($expected, $this->nn->selectLabel($output));
  }

  public function test_convTargetLabels() {
    $target = ["a","c","b","a","a","c","b","a","c"];
    $expected = [0,1,2,0,0,1,2,0,1];
   $this->assertEquals($expected, $this->nn->convTargetLabels($target));
    $labels = $this->nn->getLabels();

    $expected = ["a","c","b"];
    $this->assertEquals($expected, $labels);

    $target = [1,1,0,0];
    $expected = [0,0,1,1];
   $this->assertEquals($expected, $this->nn->convTargetLabels($target));
    $labels = $this->nn->getLabels();

    $expected = [1,0];
    $this->assertEquals($expected, $labels);

}  

public function test_activationFunction() {
  // h = [[7,-8,9], [-10,11,-12]]; 

  $expected = [[7,0,9], [0,11,0]];
 $this->assertEquals($expected, $this->nn->activationFunction($this->h));
}
public function test_activationFunctionDer() {
  // h = [[7,-8,9], [-10,11,-12]]; 

  $expected = [[1,0,1], [0,1,0]];
 $this->assertEquals($expected, $this->nn->activationFunctionDer($this->h,null));
}



public function test_train() {
  // h = [[7,-8,9], [-10,11,-12]]; 
  $features = [[0,1],[1,0],[1,1],[0,0]];
  $target = [1,1,0,0];
  $this->nn->train($features,$target,200);

  $features = [[0,1]];
  $expected = [[1]];

  $result =$this->nn->run($features);
  print_r($result);
  //$this->assertEquals($expected,$result);

  $features = [[1,0]];
  $expected = [[1]];

  $result =$this->nn->run($features);
  print_r($result);
  //$this->assertEquals($expected,$result);

  $features = [[1,1]];
  $expected = [[0]];

  $result =$this->nn->run($features);
  print_r($result);
  //$this->assertEquals($expected,$result);
  
  $features = [[0,0]];
  $expected = [[0]];

  $result =$this->nn->run($features);
  print_r($result);
 // $this->assertEquals($expected,$result);
 
 $weight_i_h = $this->nn->getWeightIH();
 //2 * 3
 print_r($weight_i_h);
 $weight_h_o = $this->nn->getWeightHO();
 //1 * 3
 print_r($weight_h_o); 

}


public function test_run() {
//   $this->markTestIncomplete(
//     'This test has not been implemented yet.'
// ); 
  $labels = ["0","1"];
  $this->nn->setLabels($labels);      
  // 2 * 3
//   array([[-1.0693864 ,  1.42897177,  0.50155784],
//   [ 1.0693864 , -0.4780464 ,  0.98885689]])
  $weight_i_h = [[-1.0693864 ,  1.42897177,  0.50155784],[1.0693864 , -0.4780464 ,  0.98885689]];
  // 3 * 1
//   array([[ 1.46712666],
//   [ 0.90174241],
//   [-0.57533633]])    
  $weight_h_o = [[1.46712666],[0.90174241],[-0.57533633]];
  $this->nn->setWeightIH($weight_i_h);
  $this->nn->setWeightHO($weight_h_o);

  $features = [[0,1]];
  $expected = [[1]];

  $result =$this->nn->run($features);
  $this->assertEquals($expected,$result);

  $features = [[1,0]];
  $expected = [[1]];

  $result =$this->nn->run($features);
  $this->assertEquals($expected,$result);

  $features = [[1,1]];
  $expected = [[0]];

  $result =$this->nn->run($features);
  $this->assertEquals($expected,$result);
  
  $features = [[0,0]];
  $expected = [[0]];

  $result =$this->nn->run($features);
  $this->assertEquals($expected,$result);      
}

}