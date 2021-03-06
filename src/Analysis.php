<?php

namespace Analysis;

 use Exception;

/**
 * Analysis class
 * ニューラルネットワーククラスの為のデータ分析用クラス
 * 
 */

class Analysis
{
    protected $epoch;

    protected $exe_time;
 
    protected $points_checker;

    protected $rates;

    protected $val_rates;
    // setter
    public function setEpoch($epoch) {
        $this->epoch = $epoch;
    }


    public function __construct($epoch)
    {
        $this->epoch = $epoch;
    }      



//学習進捗データの作成
    public function initProgressData(){
        $this->rates = [];
        $this->val_rates = [];        
        $this->points_checker = $this->epoch / 100 * 4;
        if ($this->points_checker < 10) $this->points_checker = 10;

        $this->startTimer();
    }

    public function stackLossHistory($idx,$train_loss,$val_loss,$lr){
        if (!($idx % $this->points_checker) or $idx == ($this->epoch -1)) {
            $this->rates[] =($idx).":".$train_loss.":".$lr;

            $this->val_rates[] =$val_loss;
            
            
        }        

    }

    public function getProgressData($num_input_nodes,$num_hidden_nodes,$num_output_nodes,$lr,
    $active_func_name,$labels,$lr_method){
        $this->endTimer();
        return [
            'Epochs'=>$this->epoch,
            'Learning rate'=>$lr,
            'Input neurons'=>$num_input_nodes,
            'Hidden neurons'=>$num_hidden_nodes,
            'Output neurons'=>$num_output_nodes,
            'activation_func'=>$active_func_name,
            'rates'=>$this->rates,
            'val_rates'=>$this->val_rates,            
            'point_checker'=>$this->points_checker,
            'Execution time'=>$this->exe_time,
            'Labels'=>$labels,
            'lr_method'=>$lr_method
        ];  
    }



    //実行時間計測用メソッド

    public function startTimer(){
        $this->exe_time = microtime(true);
    }

    public function endTimer(){
        $this->exe_time = round( microtime(true) - $this->exe_time ,2);
    }


}