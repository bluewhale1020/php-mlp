<?php
require 'NeuralNetwork.php';
use NeuralNetwork\NeuralNetwork;

$mlp = new NeuralNetwork(2,3,1,0.01);
$mlp->train([[0,1],[1,0],[1,1],[0,0]],[1,1,0,0],3000);

$predicted = $mlp->run([[0,1]]);
//[1]

print_r($predicted[0]);
echo"\n";
$predicted = $mlp->run([[1,1]]);
//[0]
print_r($predicted[0]);
echo"\n";
$predicted = $mlp->run([[0,0]]);
//[0]
print_r($predicted[0]);
echo"\n";
$predicted = $mlp->run([[1,0]]);
//[1]

print_r($predicted[0]);