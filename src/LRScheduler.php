<?php
namespace LRScheduler;

 use Exception;

/**
 * LRScheduler class
 * Learning rate をモデルトレーニングの進捗に合わせて逓減させる
 * 
 */

class LRScheduler
{
    protected $initial_lr;
    public $lr_method;
    protected $epoches;
    protected $decay_rate;

    public function __construct($lr_method,$initial_lr = 0.1,$epoches = 2000)
    {
        $this->lr_method = $lr_method;
        $this->initial_lr = $initial_lr;        
        $this->epoches = $epoches;
        $this->decay_rate = $this->initial_lr/$this->epoches;
    }    

 
    public function getLr($epoch= 0){

    

        switch ($this->lr_method) {
            case 'constant':
                $lr = $this->constant();
                break;
            case 'stepDecay':
                $lr =  $this->stepDecay($epoch);
                break;
            case 'timeBaseDecay':
                $lr =   $this->timeBaseDecay($epoch);
                break;
            case 'exponentialDecay':
                $lr =   $this->exponentialDecay($epoch);
                break;
            
            default:
                $lr =   $this->constant();
                break;
        }

        return $lr;

    }

    public function constant(){
        //$lr 固定
        return $this->initial_lr;
    }

    public function stepDecay($epoch){
        //epochs_dropずつ lr　を0.5倍する
        $drop = 0.5;
        $epochs_drop = 500;
        
        $lr = $this->initial_lr * pow($drop,floor(((1+$epoch)/$epochs_drop)));

        return $lr;
    }

    public function timeBaseDecay($epoch){
        //経過に応じて　lr に　低減率を掛ける 
        //lr = lr0/(1+kt)
        //lr *= (1. / (1. + decay * iterations))
        
        $lr = $this->initial_lr/(1 + $this->decay_rate * $epoch) ;

        return $lr;
    }

    public function exponentialDecay($epoch){
        //経過に応じて　lr に　指数計算した低減率を掛ける
        //lr = lr0 * e^(−kt)
        $decay = 0.0002;
        $lr = $this->initial_lr*(exp(-1 *  $decay * $epoch)) ;

        return $lr;
    }





}