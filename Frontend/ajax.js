document.getElementById('optimize-button').addEventListener('click', function() {
    fetch('api_endpoint_here', {
        method: 'POST',
        body: JSON.stringify({option: document.getElementById('option-select').value}),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result-display').innerHTML = JSON.stringify(data);
    })
    .catch(error => console.error('Error:', error));
});
