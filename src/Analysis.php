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
    protected $active_func_name;
    protected $lr;
    protected $num_input_nodes;
    protected $num_hidden_nodes;
    protected $num_output_nodes;

    protected $labels;
    protected $epoch;

    protected $exe_start_time;
    protected $exe_time;

 
    protected $points_checker;

    protected $rates;
    protected $error_lines;

    protected $progressData;

    public function __construct($num_input_nodes,$num_hidden_nodes,$num_output_nodes,$lr,
    $active_func_name,$labels,$epoch)
    {
        $this->epoch = $epoch;
        $this->lr = $lr;
        $this->num_input_nodes = $num_input_nodes;
        $this->num_hidden_nodes = $num_hidden_nodes;
        $this->num_output_nodes = $num_output_nodes;
        $this->active_func_name = $active_func_name;
        $this->labels = $labels;
    }      



//学習進捗データの作成
    public function initProgressData(){
        $this->rates = [];
        $this->points_checker = $this->epoch / 100 * 4;
        if ($this->points_checker < 10) $this->points_checker = 10;

        $this->startTimer();
    }

    public function stackLossHistory($idx,$accuracy){
        if (!($idx % $this->points_checker) or $idx == ($this->epoch -1)) {
            $this->rates[] =$accuracy;
            
            $msg = "#".($idx+1)."回目学習   ";
            $msg .="学習時誤差:".number_format($accuracy,4);
            $this->error_lines[] = $msg;

        }        

    }

    public function completeProgressData(){
        $this->endTimer();

        $this->setProgressData();

        return $this->getProgressData();
    }

    public function setProgressData(){
        $this->progressData = [
            'error_lines'=>$this->error_lines,
            'Epochs'=>$this->epoch,
            'Learning rate'=>$this->lr,
            'Input neurons'=>$this->num_input_nodes,
            'Hidden neurons'=>$this->num_hidden_nodes,
            'Output neurons'=>$this->num_output_nodes,
            'activation_func'=>$this->active_func_name,
            'rates'=>$this->rates,
            'point_checker'=>$this->points_checker,
            'Execution time'=>$this->exe_time,
            'Labels'=>$this->labels
        ];        
    }

    public function getProgressData(){
        return $this->progressData;
    }



    //実行時間計測用メソッド

    public function startTimer(){
        $this->exe_start_time = microtime(true);
    }

    public function endTimer(){
        $this->exe_time = round( microtime(true) - $this->exe_start_time ,2);
    }
    public function getExecutionTime(){
        return $this->exe_time;
    }

}