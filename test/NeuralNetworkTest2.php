<?php
require_once('vendor/autoload.php');
require 'src/NeuralNetwork.php';
require './src/DatasetManager.php';
use Dataset\DatasetManager;

class NeuralNetworkTest2 extends PHPUnit\Framework\TestCase {

    protected $nn,$reflection;
    protected $a,$b,$c,$vector;
 
    protected function setUp() {
        $this->nn = new NeuralNetwork\NeuralNetwork("sgd",3,0.1);
        $this->reflection = new \ReflectionClass($this->nn);
        $this->nn->setInputNodes(2);
        $this->nn->setOutputNodes(1);
        $this->a = [[1,2,3], [4,5,6]];
        $this->b = [[7,8,9], [10,11,12]];
        $h = [[7,-8,9], [-10,11,-12]];
        $this->h = Matrix::createFromData($h);       
        $this->c = [[13,14], [15,16], [17,18]];
        $this->vector = [3,9,2];
    }

    public function test_initLayers() {


      $method = $this->reflection->getMethod('initLayers');
      // アクセス許可
      $method->setAccessible(true);

      $method->invoke($this->nn);

      $weight_i_h = $this->nn->getWeightIH();
      //2 * 3
      // print_r($weight_i_h);
      $weight_h_o = $this->nn->getWeightHO();
      //1 * 3
      // print_r($weight_h_o);

  }

    public function test_adam_optimizer(){//$Wt,$Mt_1,$Vt_1,$grad

      // $this->markTestIncomplete(
      //   'This test has not been implemented yet.'
      // ); 

      $betaParams = ['beta1'=>0.4,'beta2'=>0.5,'beta1_pt'=>0.2,'beta2_pt'=>0.25,'ϵ'=>0];
      $this->nn->setBetaParams($betaParams);
      $this->nn->setLr(0.8);
      $wt = [[3]];$mt_1 = [[2]];$vt_1 = [[8]];$grad_t = [[4]];
          //$ϵ = 10^(−8)
        // mt = beta1 * mt_1 + (1-beta1) * grad_t
        // mt = 0.4 * [[2]] + 0.6 * [[4]] = [[3.2]]
        // vt = beta2 * vt_1 + (1-beta2) * (grad_t)^2      
        // vt = 0.5 * [[8]] + 0.5 * [[16]] = [[12]]
        // mt = mt/(1-beta1^t) 
        // mt = [[3.2]]/0.8 = [[4]]
        // vt = vt/(1-beta2^t)
        // vt = [[12]]/0.75 = [[16]]        
        // wt1 = wt - (lr / (sqrt(vt) + ϵ)) * mt
        // wt1 = [[3]] + (0.8/[[4]]) * [[4]] = [[3.8]]
        
      $method = $this->reflection->getMethod('adam_optimizer');
      // アクセス許可
      $method->setAccessible(true);
      // $wt,$mt_1,$vt_1,$grad_t
      $mawt = Matrix::createFromData($wt);
      $mamt = Matrix::createFromData($mt_1);
      $mavt = Matrix::createFromData($vt_1);
      $magd = Matrix::createFromData($grad_t);
      //return [$Wt1,$Mt,$Vt]
      list($wt1,$mt,$vt) = $method->invoke($this->nn, $mawt,$mamt,$mavt,$magd);
      print_r($wt1->toArray());
      $this->assertEquals([[4]], $mt->toArray());
      $this->assertEquals([[16]], $vt->toArray());
      $expected = [[3.799999952316284]];
     $this->assertEquals($expected, $wt1->toArray());
     $betaParams = $this->nn->getBetaParams();
     $this->assertEquals(0.4*0.2, $betaParams['beta1_pt']);
     $this->assertEquals(0.5*0.25, $betaParams['beta2_pt']);     
    }

    public function test_adjustLr() {

      // $this->markTestIncomplete(
      //   'This test has not been implemented yet.'
      // ); 

      //(lr / (sqrt(vt) + ϵ))
      $this->nn->setBetaParams(['ϵ'=>0]);      
      $vt = [[1,4,16]];
      $mavt = Matrix::createFromData($vt);
      $this->nn->setLr(2);
      $method = $this->reflection->getMethod('adjustLr');
      // アクセス許可
      $method->setAccessible(true);
    
      $expected = [[2,1,0.5]];
      $result = $method->invoke($this->nn, $mavt);
     $this->assertEquals($expected, $result->toArray());
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
    $expected = [0,2,1,0,0,2,1,0,2];
   $this->assertEquals($expected, $this->nn->convTargetLabels($target));
    $labels = $this->nn->getLabels();

    $expected = ["a","b","c"];
    $this->assertEquals($expected, $labels);
    $output = [0.1];
    $expected = ["a"];
    $this->assertEquals($expected, $this->nn->selectLabel($output));    
    $output = [0.8];
    $expected = ["b"];
    $this->assertEquals($expected, $this->nn->selectLabel($output));
    $output = [2.3];
    $expected = ["c"];
    $this->assertEquals($expected, $this->nn->selectLabel($output));

    $target = [1,1,0,0];
    $expected = [1,1,0,0];
   $this->assertEquals($expected, $this->nn->convTargetLabels($target));
    $labels = $this->nn->getLabels();

    $expected = [0,1];
    $this->assertEquals($expected, $labels);
    $output = [0.1];
    $expected = [0];
    $this->assertEquals($expected, $this->nn->selectLabel($output));    
    $output = [0.8];
    $expected = [1];
    $this->assertEquals($expected, $this->nn->selectLabel($output));
}  

public function test_activationFunctionEx() {

  $this->nn->setActiveFuncName('relu');

  $result = $this->nn->activationFunctionEx($this->h);
  $expected = [[6,0,6], [0,6,0]];
  $this->assertEquals($expected,filter_var($result->toArray(), FILTER_CALLBACK, ['options' => function ($value) {
    return intval($value);
}]));


  $this->nn->setActiveFuncName('tanh');
  $h = Matrix::createFromData([[1,-3,2], [-10,15,-4]]);  
  $result = $this->nn->activationFunctionEx($h);  
 
  $expected = [[0.7615941762924194, -0.9950547814369202, 0.9640275835990906] ,
 [-1.0, 1.0, -0.9993293285369873] ];
 $this->assertEquals($expected, $result->toArray());

}


public function test_reluFunc() {
  $this->markTestIncomplete(
    'This test has not been implemented yet.'
  );
  // h = [[7,-8,9], [-10,11,-12]]; 


  $method = $this->reflection->getMethod('reluFunc');
  // アクセス許可
  $method->setAccessible(true);

  $expected = [[6,-0.016,6], [-0.02,6,-0.024]];
 $this->assertEquals($expected, $method->invoke($this->nn, $this->h));
}

public function test_tanhFunc() {
  $this->markTestIncomplete(
    'This test has not been implemented yet.'
  );

  $method = $this->reflection->getMethod('tanhFunc');
  // アクセス許可
  $method->setAccessible(true);

  $h = [[1,-3,2], [-10,15,-4]]; 
  $expected = [[0.76159415595576, -0.99505475368673, 0.96402758007582] ,
 [-0.99999999587769, 0.99999999999981, -0.99932929973907] ];
 $this->assertEquals($expected, $method->invoke($this->nn, $h));
}

public function test_activationFunctionDerExt() {

  $this->nn->setActiveFuncName('relu');

  $result = $this->nn->activationFunctionDerExt($this->h,null);
  $expected = [[1,0,1], [0,1,0]];
  $this->assertEquals($expected,filter_var($result->toArray(), FILTER_CALLBACK, ['options' => function ($value) {
    return intval($value);
}]));


  $this->nn->setActiveFuncName('tanh');
  $fh = Matrix::createFromData([[0.7,-0.8,0.9], [-0.1,0.2,-0.3]]); 

  $expected = [[0.5099999904632568,  0.35999995470046997, 0.19000005722045898], 
  [  0.9900000095367432,  0.9599999785423279,  0.9099999666213989]];
  
  $result = $this->nn->activationFunctionDerExt(null,$fh);  
 $this->assertEquals($expected, $result->toArray());

}



public function test_reluDer() {
  $this->markTestIncomplete(
    'This test has not been implemented yet.'
  );  
  $method = $this->reflection->getMethod('reluDer');
  // アクセス許可
  $method->setAccessible(true);
  // h = [[7,-8,9], [-10,11,-12]]; 

  $expected = [[1,0.002,1], [0.002,1,0.002]];
 $this->assertEquals($expected, $method->invoke($this->nn, $this->h,null));
}
public function test_tanhDer() {
  $this->markTestIncomplete(
    'This test has not been implemented yet.'
  );  
  $method = $this->reflection->getMethod('tanhDer');
  // アクセス許可
  $method->setAccessible(true);  
  $fh = [[0.7,-0.8,0.9], [-0.1,0.2,-0.3]]; 

  $expected = [[0.51,  0.36, 0.19], [  0.99,  0.96,  0.91]];
  
 $this->assertEquals($expected, $method->invoke($this->nn, $fh));
}


public function test_train() {

  $dataset = new DatasetManager();

  $this->nn->setActiveFuncName('relu');

  // h = [[7,-8,9], [-10,11,-12]]; 
  $features = [[0,1],[1,0],[1,1],[0,0]];
  $target = [1,1,0,0];

  $dataset->setTestFeatures($features);
  $dataset->setTestTargets($target);
  $dataset->setTrainFeatures($features);
  $dataset->setTrainTargets($target);

  $this->nn->train($dataset,200,true);

  $features = [[0,1]];
  $expected = [[1]];

  $result =$this->nn->run_ex($features);
  print_r($result);
  //$this->assertEquals($expected,$result);

  $features = [[1,0]];
  $expected = [[1]];

  $result =$this->nn->run_ex($features);
  print_r($result);
  //$this->assertEquals($expected,$result);

  $features = [[1,1]];
  $expected = [[0]];

  $result =$this->nn->run_ex($features);
  print_r($result);
  //$this->assertEquals($expected,$result);
  
  $features = [[0,0]];
  $expected = [[0]];

  $result =$this->nn->run_ex($features);
  print_r($result);
 // $this->assertEquals($expected,$result);
 
 $weight_i_h = $this->nn->getWeightIH();
 //2 * 3
 print_r($weight_i_h);
 $weight_h_o = $this->nn->getWeightHO();
 //1 * 3
 print_r($weight_h_o); 

}


public function test_run_ex() {

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

  $result =$this->nn->run_ex($features);
  $this->assertEquals($expected,$result);

  $features = [[1,0]];
  $expected = [[1]];

  $result =$this->nn->run_ex($features);
  $this->assertEquals($expected,$result);

  $features = [[1,1]];
  $expected = [[0]];

  $result =$this->nn->run_ex($features);
  $this->assertEquals($expected,$result);
  
  $features = [[0,0]];
  $expected = [[0]];

  $result =$this->nn->run_ex($features);
  $this->assertEquals($expected,$result);      
}

}