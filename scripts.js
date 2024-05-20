document.getElementById('optimize-button').addEventListener('click', function() {
    const formData = new FormData(document.getElementById('features-form'));
    const inputData = Object.fromEntries(formData.entries());

    fetch('https://capstone-project-r71gc2wlr-wonderfulawsomes-projects.vercel.app/api/optimize', {  // 올바른 URL 설정
        method: 'POST',
        body: JSON.stringify(inputData),
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
