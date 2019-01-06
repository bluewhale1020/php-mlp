<?php
require_once('vendor/autoload.php');
require 'src/Calc.php';

class CalcTest extends PHPUnit\Framework\TestCase {

    protected $calc;
    protected $a,$b,$c,$vector;
 
    protected function setUp() {
        $this->calc = new Calc\Calc();
        $this->a = [[1,2,3], [4,5,6]];
        $this->b = [[7,8,9], [10,11,12]];
        $this->c = [[13,14], [15,16], [17,18]];
        $this->vector = [3,9,2];
    }

    public function test_dot() {
    //     [0] => Array
    //     (
    //         [0] => 94
    //         [1] => 100
    //     )
    // [1] => Array
    //     (
    //         [0] => 229
    //         [1] => 244
    //     )
        $expected = [[94,100],[229,244]];
       $this->assertEquals($expected, $this->calc->dot($this->a, $this->c));

    //    $this->a = [[1,2,3], [4,5,6]];
    //    $this->b = [[7,8,9], [10,11,12]];[7,10][8,11][9,12]
       
       $expected = [[50,68],[122,167]];
       $this->assertEquals($expected, $this->calc->dot($this->a, $this->b,true));       
    } 
    public function test_matrix_multiply() {
    //     [0] => Array
    //     (
    //         [0] => 7
    //         [1] => 16
    //         [2] => 27
    //     )

    // [1] => Array
    //     (
    //         [0] => 40
    //         [1] => 55
    //         [2] => 72
    //     )        
    $expected = [[7,16,27],[40,55,72]];
       $this->assertEquals($expected, $this->calc->matrix_multiply($this->a, $this->b));
       $expected = [[2,4,6], [8,10,12]];
       $this->assertEquals($expected, $this->calc->matrix_multiply(2, $this->a));

      // $this->c = [[13,14], [15,16], [17,18]];
    //    $this->vector = [3,9,2];
       $expected = [[39,42], [135,144],[34,36]];
       $this->assertEquals($expected, $this->calc->matrix_multiply($this->c,$this->vector,true));

    } 
    public function test_matrix_divide() {
        $expected = [[0.5,1,1.5], [2,2.5,3]];
       $this->assertEquals($expected, $this->calc->matrix_divide($this->a, 2));
    } 

    public function test_matrix_add() {
        // $a = [[1,2,3], [4,5,6]];
        // $b = [[7,8,9], [10,11,12]];
        // $c = [[13,14], [15,16], [17,18]];

        $expected = [[8,10,12],[14,16,18]];
       $this->assertEquals($expected, $this->calc->matrix_add($this->a, $this->b));
    }

    public function test_expand1d_2d() {
      // $this->vector = [3,9,2];
        $expected = [[3,3],[9,9],[2,2]];
       $this->assertEquals($expected, $this->calc->expand1d_2d($this->vector,2));
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
       $this->assertEquals($expected, $this->calc->initDelta($this->c));
    } 
    public function test_rand() {
    //     $expected = [];
    //    $this->assertEquals($expected, $this->calc->rand($a, $b));
       $result = $this->calc->rand(10, 20);
       print("test_rand:\n");
       print_r($result);
    } 

}