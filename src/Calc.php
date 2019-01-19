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
    public function matrix_multiply($variable,$matrix,$newaxis = false,$cut_col = false){
        //行列の要素の積を返す
        if(is_array($variable)){
            if($newaxis){//足りない要素を追加
                $rows = $this->calcMatrix->countRow($variable);
                $rows2 = $this->calcMatrix->countRow($matrix);  
                if($rows != $rows2){
                    throw new Exception("二つの行列の行数が異なります。", 1);
                    
                }             
                $columns = $this->calcMatrix->countCol($variable);
                $matrix = $this->expandColumns($matrix,$columns);

                $rows2 = $this->calcMatrix->countRow($matrix);
                if($rows != $rows2){
                    $variable = $this->expandRows($variable,$rows2);                   
                }
                
            }elseif($cut_col){//余る列をカット
                $rows = $this->calcMatrix->countRow($variable);
                $rows2 = $this->calcMatrix->countRow($matrix);  
                if($rows != $rows2){
                    throw new Exception("二つの行列の行数が異なります。", 1);
                    
                }
                $columns = $this->calcMatrix->countCol($matrix);
                $columns2 = $this->calcMatrix->countCol($variable);
                if($columns < $columns2){
                    $variable = $this->cutColumns($variable,$columns);
                }elseif($columns > $columns2){
                    $matrix = $this->cutColumns($matrix,$columns2);
                }                

            }
            
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

    protected function matrix_abs($matrix){
        return filter_var($matrix, FILTER_CALLBACK, ['options' => function ($value) {
            return abs($value);
        }]);        
    }

    public function matrix_abs_add($matrix1,$matrix2){
        //行列の絶対値の要素の和を返す
        return $this->calcMatrix->add($this->matrix_abs($matrix1),$this->matrix_abs($matrix2));
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

    public function cutColumns($matrix,$num_column){

        if(!is_array($matrix[0])){// [3,9,2];
            array_pop($matrix);
        }else{
            // [[3,9,2]];
            foreach($matrix as $key=>$row) {
                foreach ($row as $idx => $value) {
                    if($idx > ($num_column-1)){
                        unset($matrix[$key][$idx]);
                    }
                    
                }
                
             }        


        }
        return $matrix;
    }

    public function expandColumns($matrix,$num_column){

        if(!is_array($matrix[0])){// [3,9,2];
            return $this->expand1d_2d($matrix,$num_column);
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
 
    public function initWeight($cols,$rows,$active_func_name='relu'){
        //列数$cols 行数$rowsの行列を作成 データは適当な平均、偏差のランダム数
        $weightArray = [];
        

        //input_nodeと活性化関数から平均、偏差計算
        list($mean,$std) = $this->getMeanStd($rows,$active_func_name);
        for ($i=0; $i < $rows; $i++) { 
            $weightArray[$i] = null;
            for ($j=0; $j < $cols; $j++) { 
                $number = $this->gauss_ms(-1,1,$mean,$std);
                //$number = $this->rand(-1,1);
                $weightArray[$i][$j] = $number;
            }
        }
        return $weightArray;       
    }

    public function initBias($cols,$rows = 1,$init_bias = 1){
        //列数$cols 行数$rowsの行列を作成 データは適当な平均、偏差のランダム数
        $biasArray = [];

        for ($i=0; $i < $rows; $i++) { 
            $biasArray[$i] = null;
            for ($j=0; $j < $cols; $j++) { 
                $biasArray[$i][$j] = $init_bias;
            }
        }
        return $biasArray;       
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

    public function resetDelta($deltaArray){
        //$deltaArrayの要素を全て０にする
        list($rows,$cols) = $this->calcMatrix->countSize($deltaArray);

        for ($i=0; $i < $rows; $i++) { 
            for ($j=0; $j < $cols; $j++) { 
                $deltaArray[$i][$j] = 0;
            }
        }
        return $deltaArray;

    }

    public function getMeanStd($base,$active_func_name){
        //baseと活性化関数から平均、偏差計算
        if(empty($base)){
            throw new \Exception('Base should not be 0.');
           
        }
        $mean = 0;
        switch ($active_func_name) {
            case 'relu'://std * sqrt(2/n)
                $std = sqrt(2/$base);
                break;
            case 'tanh'://std * sqrt(1/n) 
                $std = sqrt(1/$base);
    
                    break;  
            case 'sigmoid'://std * sqrt(1/n) 
                $std = sqrt(1/$base);

                break;                                  
            default://mean 0, std base^-5
                $std = pow($base,-5);
                break;
        }
        
        return [$mean,$std];
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

