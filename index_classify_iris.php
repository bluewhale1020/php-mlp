<?php
ini_set('display_errors', 'On');
error_reporting(E_ALL | E_STRICT);
//制限時間変更
$default = ini_get('max_execution_time');
set_time_limit(0);
ini_set('memory_limit', '256M');
require './src/Utility.php';
require './src/DatasetManager.php';
// require 'src/TrainTestSplit.php';
use NeuralNetwork\NeuralNetwork;
use Utility\Utility;
use Dataset\DatasetManager;
use CrossValidation\TrainTestSplit;
//<< iris 問題 >>
//reluの設定
// layer:4-9-1
// lr:0.02 
// momentum:0.2
// lr_method:exponentialDecay

//adam 最適lr:$lr = 0.001;

// $input_nodes = 4;
$hidden_nodes = 9;
// $output_nodes = 1;
$lr = 0.02;
$active_func_name = 'relu';// tanh , relu , sigmoid
$mlp = new NeuralNetwork("sgd",$hidden_nodes,$lr,$active_func_name,true,0.2,true);

// $w_ih_before = $mlp->getWeightIH();
// $w_ho_before = $mlp->getWeightHO();

// $progressData = [
  // 'Epochs'=>$epoch,
  // 'Learning rate'=>$this->lr,
  // 'Input neurons'=>$this->num_input_nodes,
  // 'Hidden neurons'=>$this->num_hidden_nodes,
  // 'Output neurons'=>$this->num_output_nodes,
  // 'activation_func'=>$this->active_func_name,
  // 'rates'=>$rates,
  // 'point_checker'=>$points_checker,
  // 'Execution time'=>$execution_time
// ];
$path = "./dataset/iris.csv";

$dataset = new DatasetManager($path,true);

$split = new TrainTestSplit();

$split->run($dataset);
// $train = $result["train"];
// $test = $result["test"];
//  list($train,$test)
// $dataset->setFeatures($train[0]);
// $dataset->setTargets($train[1]);

//学習率の低減方法 lr_method
//'constant''stepDecay' 'timeBaseDecay' 'exponentialDecay'

$labels = true;
$progressData = $mlp->train($dataset,3000,$labels,"exponentialDecay");

$g_labes = $g_vals = $g_val_vals = $g_lrs = '';
$graph = $progressData['rates'];
$val_graph = $progressData['val_rates'];
$points_checker = $progressData['point_checker'];
foreach($graph as $num => $data) {
    $g_labes .= ($num*$points_checker) . ',';

    list($idx,$accuracy,$lr) = explode(":",$data);

    $msg = "#".($idx+1)."回目学習   ";
    $msg .="学習時誤差:".number_format($accuracy,4);
    $error_lines[] = $msg;
    $g_vals .= (round( $accuracy, 2)) . ',';

    $g_lrs .= $lr . ',';
}

foreach ($val_graph as $key => $value) {
  $g_val_vals .= (round( $value, 2)) . ',';
}

$g_labes = trim($g_labes, ',');
$g_vals = trim($g_vals, ',');
$g_val_vals = trim($g_val_vals, ',');
$g_lrs = trim($g_lrs, ',');


$util = new Utility();

?>

<html>
<head>
	<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/1.1.1/Chart.min.js"></script>

<!--Bootstrap４に必要なCSSとJavaScriptを読み込み-->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>

	<style>
	</style>
</head>
<body>

<br />
<br />
<hr />
<div class="container">
<div class="row">
    <div class="col-5 "  style="max-height: 500px;overflow-y:scroll;">
    <h2 class="">Progress List:</h2>  
	<ul class="list-group" style="max-width: 400px;">
<?php
  $error_lines = array_reverse($error_lines);
  foreach ($error_lines as $key => $line) {
    echo '<li class="list-group-item">';
    echo $line;
    echo "</li>";
  }
?>
	</ul>
    </div>
    </div>
    <br />
<br />
<hr />
<div class="row">
    <div class="col-4 alert alert-info">
	<ul>
	<li>Input neurons: <?= $progressData['Input neurons'] ?></li>
	<li>Hidden neurons: <?= $progressData['Hidden neurons'] ?></li>
	<li>Output neurons: <?= $progressData['Output neurons'] ?></li>
	<li>Learning rate: <?= number_format($progressData['Learning rate'],10) ?></li>
	<li>Epochs: <?= $progressData['Epochs'] ?></li>
	<li>Execution time: <?= $progressData['Execution time'] ?> sec</li>
  <li>Labels: <?php
  if($labels){
    $util->dispArray($progressData['Labels']);
  }  ?></li>

	</ul>
    </div>
    <div class="col-8">
	<h2 class="text-center">Train Loss History: <?= ucwords($progressData['activation_func']) ?></h2>

<div class="chart" style="width:600px; margin:20px auto;">
	<canvas height="200" id="lineChart" style="height:400px; margin:20px auto;"></canvas>
    </div>
	</div>


</div>

<br />
<br />
<hr />
<div class="row">
    <div class="col-4 alert alert-info" style="overflow-y:scroll;">

	<ul class="">
  <?php
    $lrsArray = explode(",",$g_lrs);
    foreach ($lrsArray as $g_lr) {
      echo '<li>';
      echo number_format($g_lr,10);
      echo "</li>";
    } 
  ?>
    </ul>

    </div>
    <div class="col-8">
	<h2 class="text-center">Learning Rate History: <?= ucwords($progressData['lr_method']) ?></h2>

<div class="chart" style="width:600px; margin:20px auto;">
	<canvas height="200" id="lineChart2" style="height:400px; margin:20px auto;"></canvas>
    </div>
	</div>


</div>


<br />
<br />
<hr />
<div class="row">
    <div class="col-4 alert alert-info" style="max-height: 500px;overflow-y:scroll;">

    <?php
	echo '<hr /><h1>Prediction:</h1>';
	$prediction = $mlp->run($dataset->getTestFeatures());
  $score = $mlp->validate($prediction,$dataset->getTestTargets());
  
  echo "<h3 class=\"text-primary \">Score:". number_format((1-$score) * 100,2)." %</h3>";
	$util->showPrediction($prediction,$dataset->getTestTargets(),true);
?>

    </div>
    <div class="col-8">
	<h2 class="text-center">Validation Loss History: <?= ucwords($progressData['activation_func']) ?></h2>

<div class="chart" style="width:600px; margin:20px auto;">
	<canvas height="200" id="lineChart3" style="height:400px; margin:20px auto;"></canvas>
    </div>
	</div>


</div>




<?php

// echo '<hr /><h1>Before</h1>';

// $util->dispMatrix( $w_ih_before,"Weight_Input_Hidden");
// echo "<br />";
// $util->dispMatrix( $w_ho_before,"Weight_Hidden_Output");
// echo "<br />";

echo '<hr /><h1>After</h1>';

$util->dispMatrix( $mlp->getWeightIH(),"Weight_Input_Hidden");
echo "<br />";
$util->dispMatrix( $mlp->getWeightHO(),"Weight_Hidden_Output");
echo "<br />";


	echo '<hr />';

?>

</div>


<script>
  $(function () {
	  
        var areaChartData = {
          labels: [<?= $g_labes ?>],
          datasets: [
            {
              fillColor: "rgba(60,141,188,0.9)",
              strokeColor: "rgba(60,141,188,0.8)",
              pointColor: "#3b8bba",
              pointStrokeColor: "rgba(60,141,188,1)",
              pointHighlightFill: "#fff",
              pointHighlightStroke: "rgba(60,141,188,1)",
              data: [<?= $g_vals ?>],
            }
          ]
        };

        var areaChartData2 = {
          labels: [<?= $g_labes ?>],
          datasets: [
            {
              fillColor: "rgba(60,141,188,0.9)",
              strokeColor: "rgba(60,141,188,0.8)",
              pointColor: "#3b8bba",
              pointStrokeColor: "rgba(60,141,188,1)",
              pointHighlightFill: "#fff",
              pointHighlightStroke: "rgba(60,141,188,1)",
              data: [<?= $g_lrs ?>],
            }
          ]
        };

        var areaChartData3 = {
          labels: [<?= $g_labes ?>],
          datasets: [
            {
              fillColor: "rgba(255,140,0,0.9)",
              strokeColor: "rgba(255,140,0,0.8)",
              pointColor: "#FF8C00",
              pointStrokeColor: "rgba(255,140,0,1)",
              pointHighlightFill: "#fff",
              pointHighlightStroke: "rgba(255,140,0,1)",
              data: [<?= $g_val_vals ?>],
            }
          ]
        };

        var areaChartOptions = {
           showScale: true,
           scaleShowGridLines: true,
           scaleGridLineColor: "rgba(0,0,0,.05)",
           scaleGridLineWidth: 1,
           scaleShowHorizontalLines: true,
           scaleShowVerticalLines: true,
           bezierCurve: true,
           bezierCurveTension: 0.3,
           pointDot: false,
           pointDotRadius: 4,
           pointDotStrokeWidth: 1,
           pointHitDetectionRadius: 20,
           datasetStroke: true,
           datasetStrokeWidth: 2,
           datasetFill: true,
           maintainAspectRatio: false,
           responsive: true,
         };
	
         var lineChartCanvas = $("#lineChart").get(0).getContext("2d");
	    var lineChart = new Chart(lineChartCanvas);
	    var lineChartOptions = areaChartOptions;
	    lineChartOptions.datasetFill = false;
	    lineChart.Line(areaChartData, lineChartOptions);

	    var lineChartCanvas2 = $("#lineChart2").get(0).getContext("2d");
	    var lineChart2 = new Chart(lineChartCanvas2);
	    lineChart2.Line(areaChartData2, lineChartOptions); 

	    var lineChartCanvas3 = $("#lineChart3").get(0).getContext("2d");
	    var lineChart3 = new Chart(lineChartCanvas3);
	    lineChart3.Line(areaChartData3, lineChartOptions);       
  });

</script>


</body>
</html>
<?php
set_time_limit($default);
?>
