<?php
namespace Dataset;


use Exception;

class DatasetManager
{

    protected $column_names=[];

    protected $features;
    protected $targets;


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
        return $this->features = $features;
    }
    public function setTargets($targets){
        return $this->targets = $targets;
    }
}