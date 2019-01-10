<?php
namespace Utility;

require 'NeuralNetwork.php';

use Exception;

class Utility
{


    public function showPrediction($prediction,$target){
        $dtable = '<dl>';
        foreach ($prediction as $key => $value) {
            if(is_array($value)){ $value = $value[0];}
            $dtable .= '<dt>Actual  [ '.$target[$key].' ]</dt>';
            $dtable .= '<dd>Predicted:    '.sprintf("%.4f",$value).'</dd>';
        }
        $dtable .= '</dl>';

        echo $dtable;
    }

    public function dispMatrix($matrix,$title = "Matrix Table"){
        $rows = count($matrix);
        $cols = count($matrix[0]);

        $table = '<table class="table text-center table-striped"><thead class="thead-light"><tr>';        

        for ($i=0; $i <= $cols; $i++) { 
            if($i == 0){
                $table .='<th></th>';
                continue;
            }
            $table .='<th scope="col">'.($i-1).'</th>';
        }

        $table .= '</tr></thead><tbody>';

        foreach ($matrix as $num_row => $row) {
            $table .= "<tr>";
            $table .= '<th scope="row">'.$num_row."</th>";
            foreach ($row as $idx => $value) {
                $table .= "<td>".sprintf("%.4f",$value)."</td>";
            }
            $table .="</tr>";
        }
        $table .="</tbody></table>";

        echo "<h4>$title</h4>";
        echo $table;

    }

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