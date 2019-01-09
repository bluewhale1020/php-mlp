<?php
ini_set('display_errors', 'On');
error_reporting(E_ALL | E_STRICT);

//require './src/NeuralNetwork.php';
require './src/Utility.php';
use NeuralNetwork\NeuralNetwork;
use Utility\Utility;

$input_nodes = 2;
$hidden_nodes = 3;
$output_nodes = 1;
$lr = 0.01;
$active_func_name = 'tanh';// tanh , relu
$mlp = new NeuralNetwork($input_nodes,$hidden_nodes,$output_nodes,$lr,$active_func_name);

$w_ih_before = $mlp->getWeightIH();
$w_ho_before = $mlp->getWeightHO();

// $progressData = [
//     'Epochs'=>$epoch,
//     'Learning rate'=>$lr,
//     'Hidden neurons'=>$this->num_hidden_nodes,
//     'rates'=>$rates,
//     'Execution time'=>$execution_time
// ];
$features =[[0,1],[1,0],[1,1],[0,0]];
$target = [1,1,0,0];

$progressData = $mlp->train($features,$target,10000);

$g_labes = $g_vals = '';
$graph = $progressData['rates'];
$points_checker = $progressData['point_checker'];
foreach($graph as $num => $val) {
    $g_labes .= ($num*$points_checker) . ',';
    $g_vals .= (round( $val, 2)) . ',';
}
$g_labes = trim($g_labes, ',');
$g_vals = trim($g_vals, ',');


$util = new Utility();

?>

<html>
<head>
	<!-- <script src="https://code.jquery.com/jquery-2.2.4.min.js" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script> -->
	<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>

	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/1.1.1/Chart.min.js"></script>
	<script src="https://code.jquery.com/jquery-2.2.4.min.js" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>

<!--Bootstrap４に必要なCSSとJavaScriptを読み込み-->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>

	<style>
		/* body { font-family: monospace; margin: 50px; }
		circle { display:none; }
		.center { text-align:center; } */
	</style>
</head>
<body>

<br />
<br />
<hr />
<div class="container">

<div class="row">
    <div class="col-4 alert alert-info">
	<ul>
	<li>Input neurons: <?= $progressData['Input neurons'] ?></li>
	<li>Hidden neurons: <?= $progressData['Hidden neurons'] ?></li>
	<li>Output neurons: <?= $progressData['Output neurons'] ?></li>
	<li>Learning rate: <?= $progressData['Learning rate'] ?></li>
	<li>Epochs: <?= $progressData['Epochs'] ?></li>
	<li>Execution time: <?= $progressData['Execution time'] ?> sec</li>
	</ul>
    </div>
    <div class="col-8">
	<h2 class="text-center">Activation function: <?= ucwords($progressData['activation_func']) ?></h2>

<div class="chart" style="width:600px; margin:20px auto;">
	<canvas height="200" id="lineChart" style="height:400px; margin:20px auto;"></canvas>
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
  });

</script>


</body>
</html>
