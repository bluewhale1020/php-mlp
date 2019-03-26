<?php
namespace Dataset;


use Exception;

class DatasetManager
{

    protected $column_names=[];
    protected $num_of_features = 0;
    protected $features;
    protected $targets;
    protected $testFeatures;
    protected $testTargets;
    protected $trainFeatures;
    protected $trainTargets;    

    public function __construct($path = null,$header = false)
    {
        if(empty($path)){
            return;
        }

        //csvファイルのデータをロード
        if(!file_exists($path)){
            throw new \Exception('The file does not exist.');
        }

        $fp = fopen($path,'r');

        if($fp ===false){
            throw new \Exception('The file can not be open.');            
        }

        $num_of_features = 0;

        if($header){
            $data = fgetcsv($fp);
            $num_of_features = count($data)-1;
            $this->column_names = array_slice($data,0,$num_of_features);
        }


        while ($data = fgetcsv( $fp )) {
            if(empty($num_of_features)){
                $num_of_features = count($data)-1;
                $this->column_names = range(0,$num_of_features-1);
            }
            $this->features[] = array_slice($data, 0, $num_of_features);
            $this->targets[] = $data[$num_of_features];
        }

        fclose($fp);

        $this->num_of_features = $num_of_features;
    }

    public function getFeatureNumber(){
        if(!empty($this->num_of_features)){
            return $this->num_of_features;
        }elseif(!empty($this->features)){
            $this->num_of_features = count($this->features[0]);
        }elseif(!empty($this->trainFeatures)){
            $this->num_of_features = count($this->trainFeatures[0]);
        }elseif(!empty($this->testFeatures)){
            $this->num_of_features = count($this->testFeatures[0]);
        }

        return $this->num_of_features;
    }

    public function getColumnName(){
        return $this->column_names;
    }
    public function getFeatures(){
        return $this->features;
    }
    public function getTargets(){
        return $this->targets;
    }
    public function setFeatures($features){
        $this->features = $features;
    }
    public function setTargets($targets){
        $this->targets = $targets;
    }

    public function getTestFeatures(){
        return $this->testFeatures;
    }
    public function getTestTargets(){
        return $this->testTargets;
    }
    public function setTestFeatures($features){
        $this->testFeatures = $features;
    }
    public function setTestTargets($targets){
        $this->testTargets = $targets;
    }  
    
    public function getTrainFeatures(){
        return $this->trainFeatures;
    }
    public function getTrainTargets(){
        return $this->trainTargets;
    }
    public function setTrainFeatures($features){
        $this->trainFeatures = $features;
    }
    public function setTrainTargets($targets){
        $this->trainTargets = $targets;
    }
    public function clearFeaturesTargets(){
        $this->features = [];
        $this->targets = [];
    }     
}