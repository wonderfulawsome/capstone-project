document.getElementById('optimize-button').addEventListener('click', function() {
    console.log("Button clicked!");
    fetch('https://capstone-project-r71gc2wlr-wonderfulawsomes-projects.vercel.app', {
        method: 'POST',
        body: JSON.stringify({
            PER: document.getElementById('PER').value,
            DividendYield: document.getElementById('DividendYield').value,
            Beta: document.getElementById('Beta').value,
            RSI: document.getElementById('RSI').value,
            Volume: document.getElementById('Volume').value,
            Volatility: document.getElementById('Volatility').value
        }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result-display').innerHTML = data.message;
    })
    .catch(error => console.error('Error:', error));
});
