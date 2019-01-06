<?php

namespace CalcMatrix;

use Exception;

class CalcMatrix{

/**
* 行数
* @param 二重配列 行列1
* @return int 行数
*/
public function countRow(array $a){
    return count($a);
}

/**
* 列数
* @param 二重配列 行列1
* @return int 列数
*/
public function countCol(array $a){
    return count($a[0]);
}

/**
* 行列のサイズ
* @param 二重配列 行列1
* @return array [行数,列数]
*/
public function countSize(array $a){
    return [$this->countRow($a), $this->countCol($a)];
}

/**
* 和
* @param 一重・二重配列 行列1
* @param 一重・二重配列 行列2
* @return 行列1 + 行列2
*/
public function add(array $a, array $b){

    if(!is_array($a[0]) and !is_array($b[0])){
        if(count($a) !== $this->count($b)){
            throw new \Exception('Different Matrix size.');
        } 
        foreach($a as $idx=>$val){
            $a[$idx] += $b[$idx];
   
        }
        return $a;              
    }

    if($this->countSize($a) !== $this->countSize($b)){
        throw new \Exception('Different Matrix size.');
    }
    foreach($a as $x=>$v){
        foreach($v as $y=>$val){
            $a[$x][$y] += $b[$x][$y];
        }
    }
    return $a;
}

/**
* 差
* @param 一重・二重配列 行列1
* @param 一重・二重配列 行列2
* @return 行列1 - 行列2
*/
public function sub(array $a, array $b){

    if(!is_array($a[0]) and !is_array($b[0])){
        if(count($a) !== $this->count($b)){
            throw new \Exception('Different Matrix size.');
        } 
        foreach($a as $idx=>$val){
            $a[$idx] -= $b[$idx];
   
        }
        return $a;              
    }


    if($this->countSize($a) !== $this->countSize($b)){
        throw new \Exception('Different Matrix size.');
    }
    foreach($a as $x=>$v){
        foreach($v as $y=>$val){
            $a[$x][$y] -= $b[$x][$y];
        }
    }
    return $a;
}

/**
* 要素の積
* @param 二重配列 行列1
* @param 二重配列 行列2
* @return 行列1 + 行列2
*/
public function elem_multiply(array $a, array $b){
    if($this->countSize($a) !== $this->countSize($b)){
        throw new \Exception('Different Matrix size.');
    }
    foreach($a as $x=>$v){
        foreach($v as $y=>$val){
            $a[$x][$y] *= $b[$x][$y];
        }
    }
    return $a;
}

/**
* 積
* @param 二重配列 行列1
* @param 二重配列 行列2
* @return 行列1 * 行列2
*/
public function multiply(array $a, array $b){
    $c = $this->countCol($b);
    $in = $this->countRow($b);
    if($this->countCol($a) !== $in){
        throw new \Exception('Invalid Matrix size. $a_cols:'.$c.'$b_rows:'.$in);
    }
    $r = $this->countRow($a);

    $ret = [];
    for($i=0;$i< $r; $i++) { $ret[$i] = array(); }
    // multiplication here
    for($ri=0;$ri<$r;$ri++) {
      for($ci=0;$ci<$c;$ci++) {
        $ret[$ri][$ci] = 0.0;
        for($j=0;$j<$in;$j++) {
          $ret[$ri][$ci] += $a[$ri][$j] * $b[$j][$ci];
        }
      }
    }    
    return $ret;
}

/**
* スカラー倍
* @param 二重配列 行列1
* @param float スカラー倍
* @return 行列1 * float
*/
public function scalar(array $a, float $m){
    array_walk_recursive($a, function(&$v, $k, $m){
        $v = $v*$m;
    }, $m);
    return $a;
}

/**
* 転置行列
* @param 二重配列 行列1
* @return 行列1の転置行列
*/
public function transpose(array $a){
    return call_user_func_array('array_map', array_merge([null], $a));
}
}