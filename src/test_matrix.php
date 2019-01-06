<?php
require 'CalcMatrix.php';
use CalcMatrix\CalcMatrix;

    // 使い方
    $matrix = new CalcMatrix();

    $a = [[1,2,3], [4,5,6]];
    $b = [[7,8,9], [10,11,12]];
    $c = [[13,14], [15,16], [17,18]];

    $sum = $matrix->add($a, $b);
    $diff = $matrix->sub($a, $b);
    $elem_product = $matrix->elem_multiply($a, $b);    
    $product = $matrix->multiply($a, $c);
    $scalar = $matrix->scalar($a, 2);
    $trans = $matrix->transpose($b);

    echo "sum:";
    print_r($sum);
    echo "diff:";
    print_r($diff);
    echo "element product:";
    print_r($elem_product);    
    echo "product:";
    print_r($product);
    echo "scalar:";
    print_r($scalar);
    echo "trans:";
    print_r($trans);

    $features = [[0,1]];
    $expected = 0;
    // 2 * 3
  //   array([[-1.0693864 ,  1.42897177,  0.50155784],
  //   [ 1.0693864 , -0.4780464 ,  0.98885689]])
    $weight_i_h = [[-1.0693864 ,  1.42897177,  0.50155784],[1.0693864 , -0.4780464 ,  0.98885689]];
    $product = $matrix->multiply($features, $weight_i_h);
    print("\n");
    print_r($product);
    $d = [[1,1]];
    $e = [[1,2,3],[0,1,2]];
    $product = $matrix->multiply($d, $e);
    print_r($product);