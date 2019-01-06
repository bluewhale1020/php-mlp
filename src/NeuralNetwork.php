<?php
namespace NeuralNetwork;
require 'Calc.php';
use Calc\Calc;


class NeuralNetwork 
{
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

    public function __construct($num_input_nodes,$num_hidden_nodes,$num_output_nodes,$lr=0.01)
    {
        //ニューラルネットワークの初期化　（インプットノード数、隠れノード数、出力ノード数、学習率）
        $this->calc = new Calc();
        $this->lr = $lr;
        $this->num_input_nodes = $num_input_nodes;
        $this->num_hidden_nodes = $num_hidden_nodes;
        $this->num_output_nodes = $num_output_nodes;

        //weight配列を作成
        $this->weight_i_h = $this->calc->initWeight($this->num_hidden_nodes,$this->num_input_nodes);
        $this->weight_h_o = $this->calc->initWeight($this->num_output_nodes,$this->num_hidden_nodes);
        
        //


    }

    public function train($features,$target,$epoch = 20,$labels = false){
        //ニューラルネットワークを教師データで学習させる
        if($labels){
            $target = $this->convTargetLabels($target);
        }else{
            $this->$labels = null;
        }

        
        $records = count($features);
        for ($idx=0; $idx < $epoch; $idx++) { 
            print("#".($idx+1)."回目学習\n");

            $this->train_network($features,$target,$records);
        }

    }

    protected function train_network($features,$target,$records){
        $delta_i_h = $this->calc->initDelta($this->weight_i_h);
        $delta_h_o = $this->calc->initDelta($this->weight_h_o);

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
            $y_error_term = $error;

            $hidden_error = $this->calc->dot($error,$this->weight_h_o,true);

            $h_error_term =$this->calc->matrix_multiply($hidden_error , $this->activation_function_der($hidden_input)); 
            //hidden_error_term = hidden_error * self.activation_function_der(hidden_outputs)
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
    }

    public function run($features){
        //ニューラルネットワークによるインプットデータの処理実行
        if (!is_array($features[0]) ){
            $features = array($features);
        }
        
        //h = W.X + b
        $hidden_input = $this->calc->dot($features,$this->weight_i_h);
        //y = f(h)
        $hidden_output = $this->activationFunction($hidden_input);

        //print_r($hidden_input);
        //h = W.X + b
        $final_input = $this->calc->dot($hidden_output,$this->weight_h_o);
        $y_hat =$final_input;
        //signals from final output layer
        //$y_hat = $this->activationFunction($final_input);


        if(!empty($this->labels)){
            return $this->selectLabel($y_hat[0]);
        }else{
            return $y_hat[0];
        }

        
    }

    public function selectLabel($output){
        //出力データ配列の数値を対応するラベルに変換して返す
        $result = [];
        foreach ($output as $key => $value) {
            $l_index = round($value);
            if(!empty($this->labels[$l_index])){
                $result[] = $this->labels[$l_index];
            }else{
                $result[] = null;
            }


        }
        return $result;
       
    }

    public function convTargetLabels($target){
        //ラベルとして与えられた教師データ(分類データ)をラベル配列に変換
        $this->labels = array_unique($target);

        //元のデータを番号データとして返す
        $target = array_map(function($x){
            return array_search($x,$this->labels);
        },$target);
        return $target;

    }

    public function activationFunction($h){
        //relu function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return max(0,$value);
        }]);
        //$y_hat = array_map("relu",$h);

        return $y_hat;

    }

    public function activation_function_der($h){
        //relu function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return ($value > 0)? 1:0;

        }]);
        //$y_hat = array_map("relu",$h);

        return $y_hat;

    }


}
