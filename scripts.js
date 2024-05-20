document.getElementById('optimize-button').addEventListener('click', function() {
    const formData = new FormData(document.getElementById('features-form'));
    const inputData = Object.fromEntries(formData.entries());

    console.log("Form data collected:", inputData);

    fetch('https://capstone-project-r71gc2wlr-wonderfulawsomes-projects.vercel.app/api/optimize', {
        method: 'POST',
        body: JSON.stringify(inputData),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok " + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);
        document.getElementById('result-display').innerHTML = JSON.stringify(data);
    })
    .catch(error => {
        console.error('Fetch error:', error);
    });
});
