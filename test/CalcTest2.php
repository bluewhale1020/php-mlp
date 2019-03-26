<?php
require_once('vendor/autoload.php');
require 'src/Calc.php';

class CalcTest extends PHPUnit\Framework\TestCase {

    protected $calc;
    protected $a,$b,$c,$vector;
 
    protected function setUp() {
       print("test");
        $this->calc = new Calc\Calc();

        $a = [[1,2,3], [4,5,6]];
        $b = [[7,8,9], [10,11,12]];
        $c = [[13,14], [15,16], [17,18]];
        $vector = [[3,9,2]];

        $this->a = Matrix::createFromData($a) ;
        $this->b = Matrix::createFromData($b) ;
        $this->c = Matrix::createFromData($c) ;
        $this->vector =  Matrix::createFromData($vector);
    }

    public function test_dot() {

        $expected = [[94,100],[229,244]];
        $result = $this->calc->dot($this->a, $this->c);
       $this->assertEquals($expected, $result->toArray());
       
       $expected = [[50,68],[122,167]];
       $result = $this->calc->dot($this->a, $this->b,true);
       $this->assertEquals($expected, $result->toArray() );       
    } 
    public function test_matrix_multiply() {
      
    $expected = [[7,16,27],[40,55,72]];
    $result =$this->calc->matrix_multiply($this->a, $this->b);
       $this->assertEquals($expected,$result->toArray() );
       $expected = [[2,4,6], [8,10,12]];
       $result =$this->calc->matrix_multiply(2, $this->a);
       $this->assertEquals($expected,$result->toArray() );

        $expected = [[39,42], [135,144],[34,36]];
       $result =$this->calc->matrix_multiply($this->c,$this->vector,true);
       $this->assertEquals($expected,$result->toArray() );

    } 
    public function test_matrix_divide() {
        $expected = [[0.5,1,1.5], [2,2.5,3]];
        $result =$this->calc->matrix_divide($this->a, 2);
       $this->assertEquals($expected,$result->toArray() );
    } 

    public function test_matrix_add() {
        $expected = [[8,10,12],[14,16,18]];
        $result =$this->calc->matrix_add($this->a, $this->b);
       $this->assertEquals($expected,$result->toArray() );
    }
    public function test_expandRows() {
      // $this->vector = [3,9,2];
        $expected = [[3,9,2],[3,9,2]];
        $result = $this->calc->expandRows($this->vector,2);
      //   print_r($result->toArray());
       $this->assertEquals($expected,$result->toArray());
    } 
    public function test_expandColumns() {
      // $this->vector = [3,9,2];
        $expected = [[3,3],[9,9],[2,2]];
        $result = $this->calc->expandColumns($this->vector->transpose(),2);
        print_r($result->toArray());
       $this->assertEquals($expected,$result->toArray());
    } 
    public function test_initWeight() {
        $expected = [];
      // $this->assertEquals($expected, $this->calc->initWeight(2, 3));
       $result = $this->calc->initWeight(2, 3);
       print("test_initWeight:\n");
       print_r($result);      
    } 
    public function test_initDelta() {
        $expected = [[0,0],[0,0],[0,0]];
        $result =$this->calc->initDelta($this->c);
       $this->assertEquals($expected,$result->toArray() );
    } 

    public function test_resetDelta() {
      $expected = [[0,0],[0,0],[0,0]];
      $result =$this->calc->resetDelta($this->c);
     $this->assertEquals($expected,$result->toArray() );
  }     
    public function test_rand() {
       $result = $this->calc->rand(10, 20);
       print("test_rand:\n");
       print_r($result);
    } 

}