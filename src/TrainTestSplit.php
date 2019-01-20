<?php


namespace CrossValidation;

use Dataset\DatasetManager;

class TrainTestSplit
{

    public function randomize_train_data($dataset,$random_state = null){

        $trainFeatures=[];
        $trainTargets=[];


        $this->seeder($random_state);        

        $features = $dataset->getTrainFeatures();
        $targets = $dataset->getTrainTargets();
        $feature_size = count($features);
        
        $random_order=[];
        
        //ランダムな順序を配列に入れる
        for($i=0; $i<$feature_size; $i++){
            array_push($random_order, $i);
        }
        shuffle($random_order);

        foreach ($random_order as $key => $idx) {
            
            $trainFeatures[] = $features[$idx];
            $trainTargets[] = $targets[$idx];                
  

        }

        return [$trainFeatures,$trainTargets];


    }

    public function run($dataset, $test_size = 0.3,$random_state = null){

        if ($test_size < 0 || $test_size >= 1) {
            throw new \Exception('test size must be between 0 and 1');
        }

        $trainFeatures=[];
        $testFeatures=[];
        $trainTargets=[];
        $testTargets=[];

        $this->seeder($random_state);        

        $features = $dataset->getFeatures();
        $targets = $dataset->getTargets();
        $feature_size = count($features);
        
        $random_order=[];
        
        //ランダムな順序を配列に入れる
        for($i=0; $i<$feature_size; $i++){
            array_push($random_order, $i);
        }
        shuffle($random_order);

        foreach ($random_order as $key => $idx) {
            
            if(count($testFeatures)/$feature_size < $test_size){//テストデータ
               $testFeatures[] = $features[$idx];
               $testTargets[] = $targets[$idx];
            }else{//学習データ
                $trainFeatures[] = $features[$idx];
                $trainTargets[] = $targets[$idx];                
            }

        }

        $dataset->setTestFeatures($testFeatures);
        $dataset->setTestTargets($testTargets);
        $dataset->setTrainFeatures($trainFeatures);
        $dataset->setTrainTargets($trainTargets);
        $dataset->clearFeaturesTargets();

        return ["train"=>[$trainFeatures,$trainTargets],"test"=>[$testFeatures,$testTargets] ];


    }

    protected function seeder($random_state = null) {
        if ($random_state === null) {
            mt_srand();
        } else {
            mt_srand($random_state);
        }
    }


}