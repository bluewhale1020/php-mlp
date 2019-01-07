<?php
ini_set('display_errors', 'On');
error_reporting(E_ALL | E_STRICT);

require './src/NeuralNetwork.php';
use NeuralNetwork\NeuralNetwork;

$input_nodes = 2;
$hidden_nodes = 3;
$output_nodes = 1;
$lr = 0.01;
$mlp = new NeuralNetwork($input_nodes,$hidden_nodes,$output_nodes,$lr);

// $progressData = [
//     'Epochs'=>$epoch,
//     'Learning rate'=>$lr,
//     'Hidden neurons'=>$this->num_hidden_nodes,
//     'rates'=>$rates,
//     'Execution time'=>$execution_time
// ];
$progressData = $mlp->train([[0,1],[1,0],[1,1],[0,0]],[1,1,0,0],3000);

$g_labes = $g_vals = '';
$graph = $progressData['rates'];
$points_checker = $progressData['point_checker'];
foreach($graph as $num => $val) {
    $g_labes .= ($num*$points_checker) . ',';
    $g_vals .= (round( $val, 2)) . ',';
}
$g_labes = trim($g_labes, ',');
$g_vals = trim($g_vals, ',');

?>

<html>
<head>
	<script src="https://code.jquery.com/jquery-2.2.4.min.js" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/1.1.1/Chart.min.js"></script>
	<style>
		body { font-family: monospace; margin: 50px; }
		circle { display:none; }
		.center { text-align:center; }
	</style>
</head>
<body>

<h2 class="center">Activation function: <?= ucwords($progressData['activation_func']) ?></h2>

<div class="chart" style="width:600px; margin:20px auto;">
	<canvas height="200" id="lineChart" style="height:400px; margin:20px auto;"></canvas>
</div>


<br />
<h3>Hidden neurons: <?= $progressData['Hidden neurons'] ?></h3>
<h3>Learning rate: <?= $progressData['Learning rate'] ?></h3>
<h3>Epochs: <?= $progressData['Epochs'] ?></h3>
<h3>Execution time: <?= $progressData['Execution time'] ?> sec</h3>

<br />
<br />





<?php
	// echo '<hr /><h1>Prediction:</h1>';
	// foreach($xor_in as $index => $input) {
	// 	$prediction = $brain->forward($input, $w1, $b1, $w2, $b2);

	// 	sm( $prediction['A'] );
	// }

	// echo '<hr /><h1>Before</h1>';
	// echo '<br /><h4>Weights matrix W1</h4>';
	// sm($w1_before);
	// echo '<br /><h4>Bias matrix B1</h4>';
	// sm($b1_before);
	// echo '<br /><h4>Weights matrix W2</h4>';
	// sm($w2_before);
	// echo '<br /><h4>Bias matrix B2</h4>';
	// sm($b2_before);
	
	// echo '<hr /><h1>After</h1>';
	// echo '<br /><h4>Weights matrix W1</h4>';
	// sm($w1);
	// echo '<br /><h4>Bias matrix B1</h4>';
	// sm($b1);
	// echo '<br /><h4>Weights matrix W2</h4>';
	// sm($w2);
	// echo '<br /><h4>Bias matrix B2</h4>';
	// sm($b2);
	// echo '<hr />';
	
	// $str  = '$w1 = $w1_before = '.var_export($w1_before, true).';' ."\n";
	// $str .= '$b1 = $b1_before = '.var_export($b1_before, true).';' ."\n";
	// $str .= '$w2 = $w2_before = '.var_export($w2_before, true).';' ."\n";
	// $str .= '$b2 = $b2_before = '.var_export($b2_before, true).';' ."\n";

	// dd($str, false);
?>




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
