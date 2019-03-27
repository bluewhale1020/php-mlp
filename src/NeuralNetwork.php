<?php
namespace NeuralNetwork;
require 'Calc.php';
use Calc\Calc;
require 'LRScheduler.php';
use LRScheduler\LRScheduler;
require 'Analysis.php';
use Analysis\Analysis;
require 'TrainTestSplit.php';
use CrossValidation\TrainTestSplit;

use \Matrix;

class NeuralNetwork 
{
    const MODELTYPE = "neuralNetwork";
    protected $optimizer;
    protected $active_func_name;
    protected $lr;
    protected $lrScheduler;
    protected $num_input_nodes;
    protected $num_hidden_nodes;
    protected $num_output_nodes;
    protected $bias;
    protected $momentum;

    protected $labels;

    protected $weight_i_h;
    protected $weight_h_o;

    protected $bias_i_h;
    protected $bias_h_o;    

    protected $calc;
    protected $trainStat;
    protected $split;

    protected $betaParams;

    protected $useExt;
    

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
    public function getBetaParams() {
        return $this->betaParams;
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
    public function setBetaParams($betaParams) {
        $this->betaParams = $betaParams;
    }
    public function setLr($lr) {
        $this->lr = $lr;
    }
    public function setInputNodes($num_input_nodes) {
        $this->num_input_nodes = $num_input_nodes;
    }
    public function setOutputNodes($num_output_nodes) {
        $this->num_output_nodes = $num_output_nodes;
    }    
    public function setActiveFuncName($active_func_name) {
        $this->active_func_name = $active_func_name;
    }    

    public function __construct($optimizer = "sgd",$num_hidden_nodes,$lr=0.01,$active_func_name = 'relu',$bias = false,$momentum = 0,$auto_detect= true)
    {
        //ニューラルネットワークの初期化　（optimizer、隠れノード数、学習率、活性化関数名[relu tanh]、
        //バイアスノードの有無、モーメンタムファクター）
        $this->calc = new Calc(null,$auto_detect);
        $this->split = new TrainTestSplit(); 
        $this->optimizer = $optimizer;       
        $this->lr = $lr;
        $this->num_hidden_nodes = $num_hidden_nodes;
        $this->active_func_name = $active_func_name;
        $this->bias = $bias;
        $this->momentum = $momentum;


        //adam hyper parameter
        if($optimizer == 'adam'){
            $this->initBetaParams();
        }
        
        if ($auto_detect and extension_loaded('matrix')) {
            $this->useExt = true;
        }else{
            $this->useExt = false;
        }        

    }

    protected function initLayers(){
        //weight配列を作成
        $this->weight_i_h = $this->calc->initWeight($this->num_hidden_nodes,$this->num_input_nodes,$this->active_func_name);
        $this->weight_h_o = $this->calc->initWeight($this->num_output_nodes,$this->num_hidden_nodes,$this->active_func_name);
        
        //bias配列作成
        if($this->bias){
            $this->bias_i_h = $this->calc->initBias($this->num_hidden_nodes);
            $this->bias_h_o = $this->calc->initBias($this->num_output_nodes);
        }
    }

        //ニューラルネットワークを教師データで学習させる
    public function train($dataset,$epoch = 2000,$labels = false,$lr_method = "constant"){

        if($labels){
            $target = $this->convTargetLabels($dataset->getTrainTargets());
            $dataset->setTrainTargets($target);
            //アウトプットノードの設定
            $this->num_output_nodes = 1;
        }else{
            $this->$labels = null;
            //アウトプットノードの設定
            $this->num_output_nodes = 1;
        }
        //インプットノードの設定
        $this->num_input_nodes = $dataset->getFeatureNumber();
        
        if(empty($this->weight_i_h)){
            //ネットワークの初期化
            $this->initLayers();
        }

        if($this->optimizer == 'sgd'){
            //LRScheduler
            $this->lrScheduler = new LRScheduler($lr_method,$this->lr,$epoch);
        }


        //学習進捗管理
        $this->trainStat = new Analysis($epoch);
        
        $records = count($dataset->getTrainFeatures());

        
        $this->onTrainStart();

        for ($idx=0; $idx < $epoch; $idx++) { 
            
            list($features,$target) = $this->onEpochStart($dataset);

            if($this->useExt){
                $train_loss = $this->train_network_ex($features,$target);

            }else{
                $train_loss = $this->train_network($features,$target,$records);

            }
                          
            $validation_loss = $this->onEpochEnd($idx,$dataset);

    
            if($train_loss < 0.07 and $validation_loss < 0.12 or $train_loss > 1000){
                $this->trainStat->setEpoch($idx+1);
                $this->trainStat->stackLossHistory($idx,$train_loss,$validation_loss,$this->lr);
                break;
            }
            $this->trainStat->stackLossHistory($idx,$train_loss,$validation_loss,$this->lr);

        }

        return $this->onTrainEnd();


    }

    protected function onTrainStart(){
        // for progressdata
        $this->trainStat->initProgressData();
        if(!empty($this->lrScheduler)){
            $this->lr = $this->lrScheduler->getLr(0);
        }
        
    }
    protected function onEpochStart($dataset){
        //学習データの順序をシャッフルする
        $result = $this->split->randomize_train_data($dataset);
        return $result;

    }
    protected function onEpochEnd($idx,$dataset){
        
        if(!empty($this->lrScheduler)){
            $this->lr = $this->lrScheduler->getLr($idx);
        }

        if($this->useExt){
            $predicted = $this->run_ex($dataset->getTestFeatures());
        }else{
            $predicted = $this->run($dataset->getTestFeatures());
        }
        

        $score = $this->validate($predicted,$dataset->getTestTargets());
        return $score;

    }

    public function validate($predicted,$targets){

        $predicted = array_merge(...$predicted);
        
        $error = [];
        if($this->labels){
            foreach ($targets as $idx => $correct) {
                if($correct == $predicted[$idx]){
                    $error[] = 0;
                }else{
                    $error[] = 1;
                }
            }
        }else{
            if($this->useExt){
                $ma_target = Matrix::createFromData([$targets]);
                $ma_predicted = Matrix::createFromData([$predicted]);
                $result = $this->calc->matrix_sub($ma_target,$ma_predicted);

                $error = $result->toArray()[0];
            }else{
                $error = $this->calc->matrix_sub($targets,$predicted);
            }

        }
        // $records = count($predicted);
        $validation_loss = $this->MSE($error);//エラー行列の各要素を二乗し、平均値を求める

        return $validation_loss;


    }

    protected function onTrainEnd(){
        if(!empty($this->lrScheduler)){
            $lr_method = $this->lrScheduler->lr_method;
        }else{
            $lr_method = "constant";
        }

        $statData = $this->trainStat->getProgressData(
            $this->num_input_nodes,$this->num_hidden_nodes,$this->num_output_nodes,
            $this->lr,$this->active_func_name,$this->labels,$lr_method);

        return $statData;        
    }

    protected function train_network_ex($features,$target){

        $weight_i_h = Matrix::createFromData($this->weight_i_h);
        $weight_h_o = Matrix::createFromData($this->weight_h_o);
        
        if($this->bias){
            $bias_h_o = Matrix::createFromData($this->bias_h_o);
            $bias_i_h = Matrix::createFromData($this->bias_i_h);                                
        }
        //デルタ配列初期化      
        $delta_i_h = Matrix::zerosLike($weight_i_h);
        $delta_h_o = Matrix::zerosLike($weight_h_o);
      
        
        if($this->optimizer == 'sgd'){
            //前回のデルタ配列（モーメンタム用） 
            $prev_i_h = $delta_i_h;
            $prev_h_o = $delta_h_o;
            $prev_b_i_h=null;
            $prev_b_h_o=null;
        }elseif($this->optimizer == 'adam'){
            // adam optimizer
            // default values of 0.9  for β1, 0.999 for β2, and 10−8 for ϵ
            $this->resetBetaParams();
            $m_i_h = $delta_i_h;
            $v_i_h = $delta_i_h;
            $m_h_o = $delta_h_o;
            $v_h_o = $delta_h_o; 
            $mb_i_h=null;$vb_i_h=null;$mb_h_o=null;$vb_h_o =null;
        }       


        $error_sum = [];
        foreach($features as $idx => $row){
            if (!is_array($row[0]) ){
                $row = array($row);
            }

            $row = Matrix::createFromData($row); 

            //h = W.X + b
            if($this->bias){
                $hidden_input = $this->calc->matrix_add($this->calc->dot($row,$weight_i_h),$bias_i_h);
            }else{
                $hidden_input = $this->calc->dot($row,$weight_i_h);
            }
            
            //y = f(h)
            $hidden_output = $this->activationFunctionEx($hidden_input);

           //h = W.X + b
            if($this->bias){
                $final_input = $this->calc->matrix_add($this->calc->dot($hidden_output,$weight_h_o),$bias_h_o);
            }else{
                $final_input = $this->calc->dot($hidden_output,$weight_h_o);
            }            
            
            //signals from final output layer
            $y_hat = $final_input;
            $y_hat = $this->activationFunctionEx($final_input);

            $y_act = $target[$idx];
            $y_act = Matrix::createFromData([[$y_act]]);
            $error = $this->calc->matrix_sub($y_act,$y_hat);

            // for accuracy check
                $error_sum[]  = $error->toArray();

            $y_error_term =$this->calc->matrix_multiply($error , $this->activationFunctionDerExt($final_input,$y_hat)); 
            //$y_error_term = $error;
            

            $hidden_error = $this->calc->dot($error,$weight_h_o,true);
            $h_error_term =$this->calc->matrix_multiply($hidden_error , $this->activationFunctionDerExt($hidden_input,$hidden_output)); 

            $delta_i_h = $this->calc->matrix_multiply($h_error_term,$row,true); 
            $delta_h_o = $this->calc->matrix_multiply($y_error_term,$hidden_output,true);

            if($this->optimizer == 'sgd'){
                $prev_i_h = $this->sgd_optimizer($weight_i_h,$delta_i_h,$prev_i_h);
                $prev_h_o = $this->sgd_optimizer($weight_h_o,$delta_h_o,$prev_h_o);
                if($this->bias){//Bh/Boバイアス配列の調整
                    $prev_b_i_h = $this->sgd_optimizer($bias_i_h,$h_error_term,$prev_b_i_h);
                    $prev_b_h_o = $this->sgd_optimizer($bias_h_o,$y_error_term,$prev_b_h_o);
                } 
            }elseif($this->optimizer == 'adam'){
                list($weight_i_h,$m_i_h,$v_i_h) = $this->adam_optimizer($weight_i_h,$m_i_h,$v_i_h,$delta_i_h);
                list($weight_h_o,$m_h_o,$v_h_o) = $this->adam_optimizer($weight_h_o,$m_h_o,$v_h_o,$delta_h_o);
                if($this->bias){//Bh/Boバイアス配列の調整
                    list($bias_i_h,$mb_i_h,$vb_i_h) = $this->adam_optimizer($bias_i_h,$mb_i_h,$vb_i_h,$h_error_term);
                    list($bias_h_o,$mb_h_o,$vb_h_o) = $this->adam_optimizer($bias_h_o,$mb_h_o,$vb_h_o,$y_error_term);
                    // $prev_b_i_h = $this->sgd_optimizer($this->bias_i_h,$h_error_term,$prev_b_i_h);
                    // $prev_b_h_o = $this->sgd_optimizer($this->bias_h_o,$y_error_term,$prev_b_h_o);
                }     
            } 

 
            //デルタ配列のリセット
            $delta_i_h = Matrix::zerosLike($weight_i_h);
            $delta_h_o = Matrix::zerosLike($weight_h_o);            
           // $delta_i_h = $this->calc->resetDelta($delta_i_h);
            // $delta_h_o = $this->calc->resetDelta($delta_h_o);            

        }

        $this->weight_h_o = $weight_h_o->toArray();
        $this->weight_i_h = $weight_i_h->toArray();

        if($this->bias){
            $this->bias_h_o = $bias_h_o->toArray();
            $this->bias_i_h = $bias_i_h->toArray();

        }

        //for accuracy check
        //$temp =$this->calc->matrix_divide($error_sum , $records);
        $accuracy = $this->MSE($error_sum);

        return $accuracy;
    }



    protected function train_network($features,$target){
        //デルタ配列初期化      
        $delta_i_h = $this->calc->initDelta($this->weight_i_h);
        $delta_h_o = $this->calc->initDelta($this->weight_h_o);

       
        
        if($this->optimizer == 'sgd'){
            //前回のデルタ配列（モーメンタム用） 
            $prev_i_h = $delta_i_h;
            $prev_h_o = $delta_h_o;
            $prev_b_i_h=null;
            $prev_b_h_o=null;
        }elseif($this->optimizer == 'adam'){
            // adam optimizer
            // default values of 0.9  for β1, 0.999 for β2, and 10−8 for ϵ
            $this->resetBetaParams();
            $m_i_h = $delta_i_h;
            $v_i_h = $delta_i_h;
            $m_h_o = $delta_h_o;
            $v_h_o = $delta_h_o; 
            $mb_i_h=null;$vb_i_h=null;$mb_h_o=null;$vb_h_o =null;
        }       


        $error_sum = [];
        foreach($features as $idx => $row){
            if (!is_array($row[0]) ){
                $row = array($row);
            }

            //h = W.X + b
            if($this->bias){
                $hidden_input = $this->calc->matrix_add($this->calc->dot($row,$this->weight_i_h),$this->bias_i_h);
            }else{
                $hidden_input = $this->calc->dot($row,$this->weight_i_h);
            }
            
            //y = f(h)
            $hidden_output = $this->activationFunction($hidden_input);

           //h = W.X + b
            if($this->bias){
                $final_input = $this->calc->matrix_add($this->calc->dot($hidden_output,$this->weight_h_o),$this->bias_h_o);
            }else{
                $final_input = $this->calc->dot($hidden_output,$this->weight_h_o);
            }            
            
            //signals from final output layer
            $y_hat = $final_input;
            $y_hat = $this->activationFunction($final_input);

            $y_act = $target[$idx];

            $error = $this->calc->matrix_sub([[$y_act]],$y_hat);

            // for accuracy check
            //if(is_null($error_sum)){
                $error_sum[]  = $error;
            // }else{
            //     $error_sum  = $this->calc->matrix_abs_add($error_sum,$error);
            // }
            /////
            $y_error_term =$this->calc->matrix_multiply($error , $this->activationFunctionDer($final_input,$y_hat)); 
            //$y_error_term = $error;
            

            $hidden_error = $this->calc->dot($error,$this->weight_h_o,true);
            $h_error_term =$this->calc->matrix_multiply($hidden_error , $this->activationFunctionDer($hidden_input,$hidden_output)); 

            $delta_i_h = $this->calc->matrix_multiply($h_error_term,$row,true); 
            $delta_h_o = $this->calc->matrix_multiply($y_error_term,$hidden_output,true);

            if($this->optimizer == 'sgd'){
                $prev_i_h = $this->sgd_optimizer($this->weight_i_h,$delta_i_h,$prev_i_h);
                $prev_h_o = $this->sgd_optimizer($this->weight_h_o,$delta_h_o,$prev_h_o);
                if($this->bias){//Bh/Boバイアス配列の調整
                    $prev_b_i_h = $this->sgd_optimizer($this->bias_i_h,$h_error_term,$prev_b_i_h);
                    $prev_b_h_o = $this->sgd_optimizer($this->bias_h_o,$y_error_term,$prev_b_h_o);
                } 
            }elseif($this->optimizer == 'adam'){
                list($this->weight_i_h,$m_i_h,$v_i_h) = $this->adam_optimizer($this->weight_i_h,$m_i_h,$v_i_h,$delta_i_h);
                list($this->weight_h_o,$m_h_o,$v_h_o) = $this->adam_optimizer($this->weight_h_o,$m_h_o,$v_h_o,$delta_h_o);
                if($this->bias){//Bh/Boバイアス配列の調整
                    list($this->bias_i_h,$mb_i_h,$vb_i_h) = $this->adam_optimizer($this->bias_i_h,$mb_i_h,$vb_i_h,$h_error_term);
                    list($this->bias_h_o,$mb_h_o,$vb_h_o) = $this->adam_optimizer($this->bias_h_o,$mb_h_o,$vb_h_o,$y_error_term);
                    // $prev_b_i_h = $this->sgd_optimizer($this->bias_i_h,$h_error_term,$prev_b_i_h);
                    // $prev_b_h_o = $this->sgd_optimizer($this->bias_h_o,$y_error_term,$prev_b_h_o);
                }     
            } 

 
            //デルタ配列のリセット
            $delta_i_h = $this->calc->resetDelta($delta_i_h);
            $delta_h_o = $this->calc->resetDelta($delta_h_o);            

        }

        // $temp = $this->calc->matrix_divide($this->calc->matrix_multiply($this->lr , $delta_h_o) , $records);
        // $this->weight_h_o = $this->calc->matrix_add($temp,$this->weight_h_o);
        // $temp = $this->calc->matrix_divide($this->calc->matrix_multiply($this->lr , $delta_i_h) , $records);
        // $this->weight_i_h = $this->calc->matrix_add($temp,$this->weight_i_h);

        //for accuracy check
        //$temp =$this->calc->matrix_divide($error_sum , $records);
        $accuracy = $this->MSE($error_sum);

        return $accuracy;
    }

    protected function sgd_optimizer(&$weight,$weight_delta,$prev_delta){
        $grad = $this->calc->matrix_multiply($this->lr , $weight_delta) ;
        //モーメンタム追加 Nesterov Momentum
        if(!empty($prev_delta)){
            $v = $this->calc->matrix_add($this->calc->matrix_multiply($this->momentum, $prev_delta),$grad);
            //v_nesterov = v + mu * (v - v_prev)   keep going, extrapolate
            $v_nest = $this->calc->matrix_add($v,$this->calc->matrix_multiply($this->momentum, $this->calc->matrix_sub($v,$prev_delta)));
            $weight = $this->calc->matrix_add($v_nest,$weight);
        }  
        return $weight_delta;      
    }

    protected function initBetaParams(){
        $this->betaParams['beta1'] = 0.9;
        $this->betaParams['beta2'] = 0.999;
        $this->betaParams['beta1_pt'] = 0.9;
        $this->betaParams['beta2_pt'] = 0.999;
        $this->betaParams['ϵ'] = 0.00000001;
                
    }
    protected function resetBetaParams(){
        $this->betaParams['beta1_pt'] = 0.9;
        $this->betaParams['beta2_pt'] = 0.999;
    }
    protected function adam_optimizer($Wt,$Mt_1,$Vt_1,$grad){

        if(empty($Mt_1) and empty($Vt_1)){
            $Vt_1 = $Mt_1 = $this->calc->initDelta($grad);      
        }
        //$ϵ = 10^(−8)
        // mt = beta1 * mt_1 + (1-beta1) * grad_t
        $temp =  $this->calc->matrix_multiply($this->betaParams['beta1'] , $Mt_1);
        $temp2 = $this->calc->matrix_multiply((1-$this->betaParams['beta1'])  , $grad);
        $Mt = $this->calc->matrix_add($temp,$temp2);

        // vt = beta2 * vt_1 + (1-beta2) * (grad_t)^2
        $temp =  $this->calc->matrix_multiply($this->betaParams['beta2'] , $Vt_1);
        $temp2 = $this->calc->matrix_multiply((1-$this->betaParams['beta2'])  , $this->calc->matrix_multiply($grad,$grad));
        $Vt = $this->calc->matrix_add($temp,$temp2);

        // mt = mt/(1-beta1^t)
        $Mt = $this->calc->matrix_divide($Mt,(1-$this->betaParams['beta1_pt'] ));

        // vt = vt/(1-beta2^t)
        $Vt = $this->calc->matrix_divide($Vt,(1-$this->betaParams['beta2_pt'] ));

        // wt1 = wt - (lr / (sqrt(vt) + ϵ)) * mt
         $temp = $this->calc->matrix_multiply($this->adjustLr($Vt),$Mt);
        $Wt = $this->calc->matrix_add($Wt,$temp);

        $this->betaParams['beta1_pt'] *= $this->betaParams['beta1'];
        $this->betaParams['beta2_pt'] *= $this->betaParams['beta2'];

        return [$Wt,$Mt,$Vt];
    }

    protected function adjustLr($Vt){

        if($this->useExt){
            return $Vt->adjustLr($this->lr,$this->betaParams['ϵ']);
        }

        //(lr / (sqrt(vt) + ϵ))
        $newLr = filter_var($Vt, FILTER_CALLBACK, ['options' => function ($value) {
            return $this->lr/(sqrt($value) + $this->betaParams['ϵ']);
        }]);   
        
        return $newLr;
    }

    protected function cutExtraElement($matrix){

        foreach ($matrix as $key => $row) {
            array_pop($matrix[$key]);
        }
        return $matrix;
    }

    protected function MSE($error){
        //エラー行列の各要素を二乗し、平均値を求める
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

    public function run_ex($features){
        $weight_i_h = Matrix::createFromData($this->weight_i_h);
        $weight_h_o = Matrix::createFromData($this->weight_h_o);

        if($this->bias){
            $bias_h_o = Matrix::createFromData($this->bias_h_o);
            $bias_i_h = Matrix::createFromData($this->bias_i_h); 

        }

        //ニューラルネットワークによるインプットデータの処理実行
        if (!is_array($features[0]) ){
            $features = array($features);
        }

        $result = [];
        foreach($features as $idx => $row){
            if (!is_array($row[0]) ){
                $row = array($row);
            }    
            
            $row = Matrix::createFromData($row);          
            //h = W.X + b
            if($this->bias){
                $hidden_input = $this->calc->matrix_add($this->calc->dot($row,$weight_i_h),$bias_i_h);
            }else{
                $hidden_input = $this->calc->dot($row,$weight_i_h);
            }
            
            //y = f(h)
            $hidden_output = $this->activationFunctionEx($hidden_input);

           //h = W.X + b
            if($this->bias){
                $final_input = $this->calc->matrix_add($this->calc->dot($hidden_output,$weight_h_o),$bias_h_o);
            }else{
                $final_input = $this->calc->dot($hidden_output,$weight_h_o);
            }            
            
            //signals from final output layer
            //$y_hat = $final_input;
            $y_hat = $this->activationFunctionEx($final_input);


            if(!empty($this->labels)){
                //$result[] =  $y_hat[0];
                $result[] = $this->selectLabel($y_hat->toArray()[0]);
            }else{
                $result[] =  $y_hat->toArray()[0];
            }
        }
        return $result;
        
    }


    public function run($features){

        if($this->useExt){
            return $this->run_ex($features);
        }

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
            if($this->bias){
                $hidden_input = $this->calc->matrix_add($this->calc->dot($row,$this->weight_i_h),$this->bias_i_h);
            }else{
                $hidden_input = $this->calc->dot($row,$this->weight_i_h);
            }
            
            //y = f(h)
            $hidden_output = $this->activationFunction($hidden_input);

           //h = W.X + b
            if($this->bias){
                $final_input = $this->calc->matrix_add($this->calc->dot($hidden_output,$this->weight_h_o),$this->bias_h_o);
            }else{
                $final_input = $this->calc->dot($hidden_output,$this->weight_h_o);
            }            
            
            //signals from final output layer
            //$y_hat = $final_input;
            $y_hat = $this->activationFunction($final_input);


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

    public function activationFunctionEx($h){//$hidden_input
        switch ($this->active_func_name) {
            case 'relu':
                return $h->reluFunc();
                break;
            case 'tanh':
                return $h->tanhFunc();
                break;  
            // case 'sigmoid':
            //     return $this->sigmoidFunc($h);
            //     break;                           
            default:
                return $h->reluFunc();
                break;
        }

    }

    public function activationFunction($h){//$hidden_input
        switch ($this->active_func_name) {
            case 'relu':
                return $this->reluFunc($h);
                break;
            case 'tanh':
                return $this->tanhFunc($h);
                break;  
            case 'sigmoid':
                return $this->sigmoidFunc($h);
                break;                           
            default:
                return $this->reluFunc($h);
                break;
        }

    }
    
    protected function reluFunc($h){
        //leaky relu function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return min(max(0.002*$value,$value), 6);
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

    protected function sigmoidFunc($h){
        //sigmoid function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return 1 / (1 + exp(-$value));
        }]);

        return $y_hat;
    }    


    public function activationFunctionDerExt($h,$fh){//$hidden_input,$hidden_output
        switch ($this->active_func_name) {
            case 'relu':
                return $h->reluDer();
                break;
            case 'tanh':
                return $fh->tanhDer();
                break;
            // case 'sigmoid':
            //     return $this->sigmoidDer($fh);
            //     break;                            
            default:
                return $h->reluDer();
                break;
        }

    }    

    public function activationFunctionDer($h,$fh){//$hidden_input,$hidden_output
        switch ($this->active_func_name) {
            case 'relu':
                return $this->reluDer($h);
                break;
            case 'tanh':
                return $this->tanhDer($fh);
                break;
            case 'sigmoid':
                return $this->sigmoidDer($fh);
                break;                            
            default:
                return $this->reluDer($h);
                break;
        }

    }

    protected function reluDer($h){

        //leaky relu der function
        $y_hat = filter_var($h, FILTER_CALLBACK, ['options' => function ($value) {
            return ($value > 0)? 1:0.002;

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
    protected function sigmoidDer($fh){
        //sigmoid der function
        // fh' = fh(1 - fh)
        $y_hat = filter_var($fh, FILTER_CALLBACK, ['options' => function ($value) {
            return $value*(1.0 - $value);
        }]);

        return $y_hat;
    }



    
}
