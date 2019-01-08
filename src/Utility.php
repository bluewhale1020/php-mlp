<?php
namespace Utility;

require 'NeuralNetwork.php';
//use NeuralNetwork\NeuralNetwork;
use Exception;

class Utility
{
    public function saveModel($model,$path){
        //学習済みモデルの保存
        $modelData = serialize($model);
        if(!file_exists($path)){
            file_put_contents($path,$modelData);
            return true;
        }else{
            throw new \Exception('The same filename already exists.');
        }

    }

    public function loadModel($path){
        //保存したモデルをロード
        if(file_exists($path)){
            $modelData = file_get_contents($path);
            $model = unserialize($modelData);
            if(!empty($model)){
                return $model;
            }
        }else{
            throw new \Exception('The file does not exist.');
        }
        

    }
    

}