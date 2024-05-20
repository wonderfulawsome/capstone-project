document.getElementById('features-form').addEventListener('submit', function(event) {
    event.preventDefault();  // 폼의 기본 제출 막기
    const formData = new FormData(this);

    fetch('https://capstone-project-r71gc2wlr-wonderfulawsomes-projects.vercel.app', {
        method: 'POST',
        body: JSON.stringify(Object.fromEntries(formData)),
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
