<?php
ini_set('display_errors', 'On');
error_reporting(E_ALL | E_STRICT);

require './src/Utility.php';
use NeuralNetwork\NeuralNetwork;
use Utility\Utility;

//<< xor 問題 >>
//tanhの最適設定
// layer:2-3-1
// lr:0.2 
// momentum:0.3  0.4以上でweightの値が発散するリスク
//sigmoidの最適設定
// layer:2-3-1
// lr:0.2 
// momentum:0.5
//reluの最適設定
// layer:2-3-1
// lr:0.05 
// momentum:0.2

$input_nodes = 2;
$hidden_nodes = 3;
$output_nodes = 1;
$lr = 0.05 ;
$active_func_name = 'relu';// tanh , relu , sigmoid
$mlp = new NeuralNetwork($input_nodes,$hidden_nodes,$output_nodes,$lr,
$active_func_name,true,0.2);

$w_ih_before = $mlp->getWeightIH();
$w_ho_before = $mlp->getWeightHO();



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
$features =[[0,1],[1,0],[1,1],[0,0]];
$target = [1,1,0,0];
//学習率の低減方法 lr_method
//'constant''stepDecay' 'timeBaseDecay' 'exponentialDecay'

$progressData = $mlp->train($features,$target,30000,false,"exponentialDecay");

$g_labes = $g_vals = $g_lrs = '';
$graph = $progressData['rates'];
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
$g_labes = trim($g_labes, ',');
$g_vals = trim($g_vals, ',');
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
	</ul>
    </div>
    <div class="col-8">
	<h2 class="text-center">Loss History: <?= ucwords($progressData['activation_func']) ?></h2>

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

<?php
	echo '<hr /><h1>Prediction:</h1>';
	$prediction = $mlp->run($features);

	$util->showPrediction($prediction,$target);

echo '<hr /><h1>Before</h1>';

$util->dispMatrix( $w_ih_before,"Weight_Input_Hidden");
echo "<br />";
$util->dispMatrix( $w_ho_before,"Weight_Hidden_Output");
echo "<br />";

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
  });

</script>


</body>
</html>
