<?php
namespace NeuralNetwork;
require 'Calc.php';
use Calc\Calc;


class NeuralNetwork 
{
    const MODELTYPE = "neuralNetwork";
    protected $active_func_name;
    protected $lr;
    protected $num_input_nodes;
    protected $num_hidden_nodes;
    protected $num_output_nodes;

    protected $labels;

    protected $weight_i_h;
    protected $weight_h_o;

    protected $calc;

    // getter
    public function getWeightIH() {
        return $this->weight_i_h;
    }
    public function getWeightHO() {
        return $this->weight_h_o;
    }
    public function getLabels() {
        return $this->labels;
    }
    public function getLr() {
        return $this->lr;
    }
    public function getInputNodes() {
        return $this->num_input_nodes;
    }
    public function getHiddenNodes() {
        return $this->num_hidden_nodes;
    }
    public function getOutputNodes() {
        return $this->num_output_nodes;
    }
    // setter
    public function setWeightIH($W) {
        $this->weight_i_h = $W;
    }
    public function setWeightHO($W) {
        $this->weight_h_o = $W;
    }
    public function setLabels($labels) {
        $this->labels = $labels;
    }

    public function __construct($num_input_nodes,$num_hidden_nodes,$num_output_nodes,$lr=0.01,$active_func_name = 'relu')
    {
        //ニューラルネットワークの初期化　（インプットノード数、隠れノード数、出力ノード数、学習率、活性化関数名[relu tanh]）
        $this->calc = new Calc();
        $this->lr = $lr;
        $this->num_input_nodes = $num_input_nodes;
        $this->num_hidden_nodes = $num_hidden_nodes;
        $this->num_output_nodes = $num_output_nodes;
        $this->active_func_name = $active_func_name;

        //weight配列を作成
        $this->weight_i_h = $this->calc->initWeight($this->num_hidden_nodes,$this->num_input_nodes,$active_func_name);
        $this->weight_h_o = $this->calc->initWeight($this->num_output_nodes,$this->num_hidden_nodes,$active_func_name);
        
        //


    }

    public function train($features,$target,$epoch = 2000,$labels = false){
        //ニューラルネットワークを教師データで学習させる
        if($labels){
            $target = $this->convTargetLabels($target);
        }else{
            $this->$labels = null;
        }

        
        $records = count($features);

        // for progressdata
        $rates = [];
        $points_checker = $epoch / 100 * 4;
        if ($points_checker < 10) $points_checker = 10;
        ///

        $execution_start_time = microtime(true);


        for ($idx=0; $idx < $epoch; $idx++) { 
            

            $accuracy = $this->train_network($features,$target,$records);

            if (!($idx % $points_checker) or $idx == ($epoch -1)) {
                print("#".($idx+1)."回目学習   ");
                $rates[] =$accuracy;
                
                print("学習時誤差:".number_format($accuracy,4)."\n");
                print("<br />");

            }
                                    
        }

        $execution_time = round( microtime(true) - $execution_start_time ,2);

        $progressData = [
            'Epochs'=>$epoch,
            'Learning rate'=>$this->lr,
            'Input neurons'=>$this->num_input_nodes,
            'Hidden neurons'=>$this->num_hidden_nodes,
            'Output neurons'=>$this->num_output_nodes,
            'activation_func'=>$this->active_func_name,
            'rates'=>$rates,
            'point_checker'=>$points_checker,
            'Execution time'=>$execution_time,
            'Labels'=>$this->labels
        ];

        return $progressData;
    }

    protected function train_network($features,$target,$records){
        $delta_i_h = $this->calc->initDelta($this->weight_i_h);
        $delta_h_o = $this->calc->initDelta($this->weight_h_o);
        $error_sum = null;
        foreach($features as $idx => $row){
            if (!is_array($row[0]) ){
                $row = array($row);
            }
    

            //h = W.X + b
            $hidden_input = $this->calc->dot($row,$this->weight_i_h);
            //y = f(h)
            $hidden_output = $this->activationFunction($hidden_input);

            
            //h = W.X + b
            $final_input = $this->calc->dot($hidden_output,$this->weight_h_o);
            //signals from final output layer
            $y_hat = $final_input;
            //$y_hat = $this->activationFunction($final_input);

            $y_act = $target[$idx];

            $error = $this->calc->matrix_sub([[$y_act]],$y_hat);

            // for accuracy check
            if(is_null($error_sum)){
                $error_sum  = $error;
            }else{
                $error_sum  = $this->calc->matrix_add($error_sum,$error);
            }
            /////

            $y_error_term = $error;

            $hidden_error = $this->calc->dot($error,$this->weight_h_o,true);

            //hidden_error_term = hidden_error * self.activationFunctionDer(hidden_outputs)
            $h_error_term =$this->calc->matrix_multiply($hidden_error , $this->activationFunctionDer($hidden_input,$hidden_output)); 
            
            //print("h_error_term + row:");
            //print_r($h_error_term);print_r($row);
            $temp = $this->calc->matrix_multiply($h_error_term,$row,true); 
            $delta_i_h = $this->calc->matrix_add($temp,$delta_i_h);
            
            $temp = $this->calc->matrix_multiply($y_error_term,$hidden_output,true); 
            $delta_h_o = $this->calc->matrix_add($temp,$delta_h_o);

        }

        $temp = $this->calc->matrix_divide($this->calc->matrix_multiply($this->lr , $delta_h_o) , $records);
        $this->weight_h_o = $this->calc->matrix_add($temp,$this->weight_h_o);
        $temp = $this->calc->matrix_divide($this->calc->matrix_multiply($this->lr , $delta_i_h) , $records);
        $this->weight_i_h = $this->calc->matrix_add($temp,$this->weight_i_h);


        //for accuracy check
        $temp =$this->calc->matrix_divide($error_sum , $records);
        $accuracy = $this->MSE(($temp));

        return $accuracy;
    }

    protected function MSE($error){
        //エラー行列の各要素を二乗し、平均で割る
        $serror = [];
        $total =0;
        array_walk_recursive($error, function ($value) use (&$serror) {
            $serror[] = $value*$value;
        });
        $total = count($serror);
        if(!empty($total)){
            return (array_sum($serror)/$total);
        }else{
            return null;
        }
    }

    public function run($features){
        //ニューラルネットワークによるインプットデータの処理実行
        if (!is_array($features[0]) ){
            $features = array($features);
        }

        $result = [];
        foreach($features as $idx => $row){
            if (!is_array($row[0]) ){
                $row = array($row);
            }       
            //h = W.X + b
            $hidden_input = $this->calc->dot($row,$this->weight_i_h);
            //y = f(h)
            $hidden_output = $this->activationFunction($hidden_input);

            //print_r($hidden_input);
            //h = W.X + b
            $final_input = $this->calc->dot($hidden_output,$this->weight_h_o);
            $y_hat =$final_input;
            //signals from final output layer
            //$y_hat = $this->activationFunction($final_input);


            if(!empty($this->labels)){
                //$result[] =  $y_hat[0];
                $result[] = $this->selectLabel($y_hat[0]);
            }else{
                $result[] =  $y_hat[0];
            }
        }
        return $result;
        
    }

    public function selectLabel($output){
        //出力データ配列の数値を対応するラベルに変換して返す
        $result = [];

        $maxVal=0;
        if(count($output) ==1){//output node が　１
            $l_index = round($output[0]);
            if(isset($this->labels[$l_index])){
                $result[] = $this->labels[$l_index];
            }else{
                $result[] =  null;
            }         
        }else{//複数のoutput node
            $maxs    = array_keys($output, max($output));
            $key_max = $maxs[0]; 
            if(isset($this->labels[$key_max])){
                $result[] = $this->labels[$key_max];
            }else{
                $result[] =  null;
            }                       
        }

        return $result;
       
    }

    public function convTargetLabels($target){
        //ラベルとして与えられた教師データ(分類データ)をラベル配列に変換
        $temp = array_unique($target);
        sort($temp);
        $this->labels = array_values($temp);

        //元のデータを番号データとして返す
        $target = array_map(function($x){
            return array_search($x,$this->labels);
        },$target);
        return $target;

    }

    public function activationFunction($h){//$hidden_input
        switch ($this->active_func_name) {
            case 'relu':
                return $this->reluFunc($h);
                break;
            case 'tanh':
                return $this->tanhFunc($h);
                break;            
            default:
                return $this->reluFunc($h);
                break;
        }

    }
    
    protected function reluFunc($h){
        //relu function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return max(0,$value);
        }]);

        return $y_hat;
    }

    protected function tanhFunc($h){
        //tanh function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return tanh($value);
        }]);

        return $y_hat;
    }

    public function activationFunctionDer($h,$fh){//$hidden_input,$hidden_output
        switch ($this->active_func_name) {
            case 'relu':
                return $this->reluDer($h);
                break;
            case 'tanh':
                return $this->tanhDer($fh);
                break;            
            default:
                return $this->reluDer($h);
                break;
        }

    }

    protected function reluDer($h){
        //relu der function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return ($value > 0)? 1:0;

        }]);

        return $y_hat;
    }
    protected function tanhDer($fh){
        //tanh der function
        // fh' = 1 - fh**2
        $y_hat = filter_var($fh, FILTER_CALLBACK, ['options' => function ($value) {
            return (1 - $value*$value);
        }]);

        return $y_hat;
    }

}
