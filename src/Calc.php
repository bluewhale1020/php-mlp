<?php
namespace Calc;


 require 'CalcMatrix.php';
 use CalcMatrix\CalcMatrix;
 use Exception;

class Calc
{
    public $calcMatrix;

    public function __construct($calcMatrix = null)
    {
        if(is_null($calcMatrix)){
            $this->calcMatrix = new CalcMatrix();
        }else{
            $this->calcMatrix = $calcMatrix;
        }
        
    }

    public function dot($matrix1,$matrix2,$T = false){
        //行列の内積を返す
        if($T){
            $matrix2 =  $this->calcMatrix->transpose($matrix2);
        }        
        return $this->calcMatrix->multiply($matrix1,$matrix2);
    }
    public function matrix_multiply($variable,$matrix,$newaxis = false){
        //行列の要素の積を返す
        if(is_array($variable)){
            if($newaxis){
                $rows = $this->calcMatrix->countRow($variable);
                $rows2 = $this->calcMatrix->countRow($matrix);  
                if($rows != $rows2){
                    throw new Exception("二つの行列の列数が異なります。", 1);
                    
                }             
                $columns = $this->calcMatrix->countCol($variable);
                $matrix = $this->expandColumns($matrix,$columns);

                $rows2 = $this->calcMatrix->countRow($matrix);
                if($rows != $rows2){
                    $variable = $this->expandRows($variable,$rows2);                   
                }
                
            }
            //print_r($variable);print_r($matrix);
            return $this->calcMatrix->elem_multiply($variable,$matrix); 

        }else{
            return $this->calcMatrix->scalar($matrix,$variable);
        }

      
    }
    public function matrix_divide($matrix,$denominator){
        //行列の要素を$denominatorで割った結果を返す
        return $this->calcMatrix->scalar($matrix,1/$denominator);
    }
    public function matrix_add($matrix1,$matrix2){
        //行列の要素の和を返す
        return $this->calcMatrix->add($matrix1,$matrix2);
    }

    public function matrix_sub($matrix1,$matrix2){
        //行列の要素の差を返す
        return $this->calcMatrix->sub($matrix1,$matrix2);
    }    

    public function expandRows($matrix,$num_row){
        $newmatrix = [];       
            for ($i=0; $i < $num_row; $i++) { 
                
                $newmatrix[] = $matrix[0];
            }           

        return $newmatrix;
    }

    public function expandColumns($matrix,$num_column){

        if(!is_array($matrix[0])){// [3,9,2];
            return expand1d_2d($matrix,$num_column);
        }elseif(count($matrix) == 1){
            // [[3,9,2]];
            return $this->expand1d_2d($matrix[0],$num_column);
        }else{
            throw new \Exception('expandColumns:can not expand $matrix due to its size.');
        }

 
    }

    public function expand1d_2d($matrix,$num_column){
        $newmatrix = [];
        $row = [];        
        foreach ($matrix as $key => $value) {
            for ($i=0; $i < $num_column; $i++) { 
                
                $row[] = $value;
            }           
            $newmatrix[] = $row;
            $row = [];
        }


        return $newmatrix;
    }
 
    public function initWeight($cols,$rows){
        //列数$cols 行数$rowsの行列を作成 データは平均０、偏差0.2のランダム数
        $weightArray = [];

        for ($i=0; $i < $rows; $i++) { 
            $weightArray[$i] = null;
            for ($j=0; $j < $cols; $j++) { 
                $number = $this->gauss_ms(-1,1,0.2);
                //$number = $this->rand(-1,1);
                $weightArray[$i][$j] = $number;
            }
        }
        return $weightArray;       
    }
    public function initDelta($weightArray){
        //$weightArrayと同型で、要素０の行列を返す
        list($rows,$cols) = $this->calcMatrix->countSize($weightArray);
        $deltaArray = [];

        for ($i=0; $i < $rows; $i++) { 
            $deltaArray[$i] = null;
            for ($j=0; $j < $cols; $j++) { 
                $deltaArray[$i][$j] = 0;
            }
        }
        return $deltaArray;

    }   
    
     /**
     * Generate random float number.
     *
     * @param float|int $min
     * @param float|int $max
     * @return float
     */
    public function rand($min = 0, $max = 1)
    {
        return ($min + ($max - $min) * (mt_rand() / mt_getrandmax()));
    }

    function gauss()
    { // N(0,1)
    // returns random number with normal distribution:
    // mean=0
    // std dev=1
    
    // auxilary vars
    $x=$this->random_0_1();
    $y=$this->random_0_1();
    
    // two independent variables with normal distribution N(0,1)
    $u=sqrt(-2*log($x))*cos(2*pi()*$y);
    $v=sqrt(-2*log($x))*sin(2*pi()*$y);
    
    // i will return only one, couse only one needed
    return $u;
    }

    function random_0_1()
    { // auxiliary function
    // returns random number with flat distribution from 0 to 1
    return (float)rand()/(float)getrandmax();
    }    
    function gauss_ms($min,$max,$m=0.0,$s=1.0)
    { // N(m,s)
        // returns random number with normal distribution:
        // mean=m
        // std dev=s
        $randomNumber = $this->gauss()*$s+$m;    
        while ($randomNumber < $min or $randomNumber > $max) {
            $randomNumber = $this->gauss()*$s+$m;

        }
        return $randomNumber;
    }     
}

