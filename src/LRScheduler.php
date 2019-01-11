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

    protected $decay;

    public function __construct($initial_lr = 0.1,$epoches = 2000)
    {
        $this->initial_lr = $initial_lr;        
        $this->decay = $initial_lr/$epoches;
    }    

    public function constant(){
        //$lr 固定
        return $this->initial_lr;
    }

    public function stepDecay($epoch){
        //epochs_dropずつ lr　を0.5倍する
        $drop = 0.5;
        $epochs_drop = 10;
        
        $lr = $this->initial_lr * pow($drop,floor(((1+$epoch)/$epochs_drop)));

        return $lr;
    }

    public function timeBaseDecay($epoch){
        //経過に応じて　lr に　低減率を掛ける
        //lr *= (1. / (1. + decay * iterations))

        $lr = $this->initial_lr/(1 + $this->decay * $epoch) ;

        return $lr;
    }

    public function exponentialDecay($epoch){
        //経過に応じて　lr に　指数計算した低減率を掛ける
        //lr = lr0 * e^(−kt)

        $lr = $this->initial_lr*(exp(-1 * $this->decay * $epoch)) ;

        return $lr;
    }

}