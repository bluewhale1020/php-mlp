<?php

namespace CalcMatrix;

use Exception;

class CalcMatrixExt{

/**
* 行数
* @param 二重配列 行列1
* @return int 行数
*/
public function countRow( $a){
    return ($a->shape())[0];
}

/**
* 列数
* @param 二重配列 行列1
* @return int 列数
*/
public function countCol( $a){
    return ($a->shape())[1];
}

/**
* 行列のサイズ
* @param 二重配列 行列1
* @return array [行数,列数]
*/
public function countSize( $a){
    return $a->shape();
}

/**
* 和
* @param 一重・二重配列 行列1
* @param 一重・二重配列 行列2
* @return 行列1 + 行列2
*/
public function add( $a,  $b){


    if($this->countSize($a) !== $this->countSize($b)){
        throw new \Exception('Different Matrix size.');
    }

    return $a->plus($b);
}

/**
* 差
* @param 一重・二重配列 行列1
* @param 一重・二重配列 行列2
* @return 行列1 - 行列2
*/
public function sub( $a,  $b){

    if($this->countSize($a) !== $this->countSize($b)){
        throw new \Exception('Different Matrix size.');
    }

    return $a->minus($b);
}

/**
* 要素の積
* @param 二重配列 行列1
* @param 二重配列 行列2
* @return 行列1 + 行列2
*/
public function elem_multiply( $a,  $b){
    if($this->countSize($a) !== $this->countSize($b)){
        throw new \Exception('Different Matrix size.');
    }
    
    return $a->componentwiseProd($b);
}

/**
* 積
* @param 二重配列 行列1
* @param 二重配列 行列2
* @return 行列1 * 行列2
*/
public function multiply( $a,  $b){
    $c = $this->countCol($b);
    $in = $this->countRow($b);
    if($this->countCol($a) !== $in){
        throw new \Exception('Invalid Matrix size. $a_cols:'.$c.'$b_rows:'.$in);
    }
   
    return $a->mul($b);
}

/**
* スカラー倍
* @param 二重配列 行列1
* @param float スカラー倍
* @return 行列1 * float
*/
public function scalar( $a, float $m){
    return $a->scale($m);
}

/**
* 転置行列
* @param 二重配列 行列1
* @return 行列1の転置行列
*/
public function transpose( $a){
    return $a->transpose();
}
}