<?php
ini_set('memory_limit', '256M');


require 'CalcMatrix.php';
use CalcMatrix\CalcMatrix;

    $matrix = new CalcMatrix();

// $a = [[1,2,3], [4,5,6]];
// $b = [[7,8,9], [10,11,12]];
// $c = [[13,14], [15,16], [17,18]];

$SIZE = 50;

// PHPの行列
$a = random_array($SIZE, $SIZE);
$b = random_array($SIZE, $SIZE);
$c = random_array($SIZE, $SIZE);


// test php_matrix

$matA = Matrix::createFromData($a);

$matB = Matrix::createFromData($b);

$matC = Matrix::createFromData($c);

$time_start = microtime(true);

print("[和]\n");
for ($i = 0; $i < 100; $i++) {

    $matSum = $matA->plus($matB);
   
}
$time = microtime(true) - $time_start;
echo "php_matrixを使った場合 => {$time}秒\n";


// test CalcMatrix
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $sum = $matrix->add($a, $b);

}
$time2 = microtime(true) - $time_start;
echo "素のPHP版CalcMatrix => {$time2}秒\n";
$rate = round(($time2 / $time),2);
echo "速度比率 calcmatrix / php_matrix = {$rate}倍\n";
echo "\n";

print("[差]\n");
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {

    $matDiff = $matA->minus($matB);
 
}
$time = microtime(true) - $time_start;
echo "php_matrixを使った場合 => {$time}秒\n";

// test CalcMatrix
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {

    $diff = $matrix->sub($a, $b);

}
$time2 = microtime(true) - $time_start;
echo "素のPHP版CalcMatrix => {$time2}秒\n";
$rate = round(($time2 / $time),2);
echo "速度比率 calcmatrix / php_matrix = {$rate}倍\n";
echo "\n";

print("[要素同士の積]\n");
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {

    $matElemProduct = $matA->componentwiseProd($matB);
   
}
$time = microtime(true) - $time_start;
echo "php_matrixを使った場合 => {$time}秒\n";


// test CalcMatrix
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $elem_product = $matrix->elem_multiply($a, $b);    

}
$time2 = microtime(true) - $time_start;
echo "素のPHP版CalcMatrix => {$time2}秒\n";
$rate = round(($time2 / $time),2);
echo "速度比率 calcmatrix / php_matrix = {$rate}倍\n";
echo "\n";

print("[積]\n");
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $matD = $matA->mul($matC);  
}
$time = microtime(true) - $time_start;
echo "php_matrixを使った場合 => {$time}秒\n";


// test CalcMatrix
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    
    $product = $matrix->multiply($a, $c);

}
$time2 = microtime(true) - $time_start;
echo "素のPHP版CalcMatrix => {$time2}秒\n";
$rate = round(($time2 / $time),2);
echo "速度比率 calcmatrix / php_matrix = {$rate}倍\n";
echo "\n";

print("[スカラー倍]\n");
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {

    $matScalar = $matA->scale(2);
    
}
$time = microtime(true) - $time_start;
echo "php_matrixを使った場合 => {$time}秒\n";


// test CalcMatrix
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {

    $scalar = $matrix->scalar($a, 2);

}
$time2 = microtime(true) - $time_start;
echo "素のPHP版CalcMatrix => {$time2}秒\n";
$rate = round(($time2 / $time),2);
echo "速度比率 calcmatrix / php_matrix = {$rate}倍\n";
echo "\n";

print("[転置]\n");
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $matTrans = $matA->transpose();    
}
$time = microtime(true) - $time_start;
echo "php_matrixを使った場合 => {$time}秒\n";


// test CalcMatrix
$time_start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $trans = $matrix->transpose($b);
}
$time2 = microtime(true) - $time_start;
echo "素のPHP版CalcMatrix => {$time2}秒\n";
$rate = round(($time2 / $time),2);
echo "速度比率 calcmatrix / php_matrix = {$rate}倍\n";
echo "\n";

    //$expected = [[94,100],[229,244]];
    // $matD = $matA->mul($matC);
    // print("積：");        
    // print_r($matD->toArray()) ;

    // $expected = [[7,16,27],[40,55,72]];
    // $matElemProduct = $matA->componentwiseProd($matB);
    // print("要素同士の積：");    
    // print_r($expected);
    // print_r($matElemProduct->toArray());

    // $expected = [[8,10,12],[14,16,18]];
    // $matSum = $matA->plus($matB);
    // print("和：");    
    // print_r($expected);    
    // print_r($matSum->toArray());

    // $expected = [[-6,-6,-6],[-6,-6,-6]];
    // $matDiff = $matA->minus($matB);
    // print("差：");    
    // print_r($expected);    
    // print_r($matDiff->toArray());

    // $expected = [[2,4,6],[8,10,12]];
    // $matScalar = $matA->scale(doubleval(2));
    // print("スカラー倍：");    
    // print_r($expected);    
    // print_r($matScalar->toArray());    

    // $expected = [[1,4],[2,5],[3,6]];
    // $matTrans = $matA->transpose();
    // print("転置：");
    // print_r($expected);    
    // print_r($matTrans->toArray());


    function random_array($r, $c) {
        $result = [];
        for ($i = 0; $i < $r; $i++) {
            $result[$i] = array_fill(0, $c, 0);
        }
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $result[$i][$j] = rand(0, 10000) / 10000.0; 
            }
        }
        return $result;
    }