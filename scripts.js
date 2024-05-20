document.getElementById('optimize-button').addEventListener('click', function() {
    // 폼 데이터를 가져옵니다.
    const formData = new FormData(document.getElementById('features-form'));
    const inputData = Object.fromEntries(formData.entries());

    console.log("Form data collected:", inputData);  // 폼 데이터 로그 출력

    // 서버에 데이터 전송
    fetch('https://capstone-project-r71gc2wlr-wonderfulawsomes-projects.vercel.app/api/optimize', {
        method: 'POST',
        body: JSON.stringify(inputData),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log("Server response received:", response);  // 서버 응답 로그 출력
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);  // 응답 데이터 로그 출력
        // 결과를 화면에 표시
        document.getElementById('result-display').innerHTML = JSON.stringify(data);
    })
    .catch(error => {
        console.error('Error:', error);  // 에러 로그 출력
    });
});
