<?php
// Path to the image file to send
$image_path = 'dick_rater_test_images_input/1.jpg';

// Create a new cURL resource
$ch = curl_init();

// Set the URL of the FastAPI endpoint
curl_setopt($ch, CURLOPT_URL, 'e709-107-222-215-224.ngrok.io/predict');

// Set the HTTP method to POST
curl_setopt($ch, CURLOPT_POST, true);

// Set the file as the request body
curl_setopt($ch, CURLOPT_POSTFIELDS, ['file' => new CurlFile($image_path)]);

// Receive the response as a string
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

// Send the request and get the response
$response = curl_exec($ch);

// Close the cURL resource
curl_close($ch);

// Output the response
echo $response;
?>