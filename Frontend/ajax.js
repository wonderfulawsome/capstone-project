document.getElementById('optimize-button').addEventListener('click', function() {
    fetch('https://capstone-project-r71gc2wlr-wonderfulawsomes-projects.vercel.app/api/submit_data', {
        method: 'POST',
        body: JSON.stringify({option: document.getElementById('option-select').value}),
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

