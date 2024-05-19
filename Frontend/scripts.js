// scripts.js
document.getElementById('option-select').addEventListener('change', function() {
    let selectedOption = this.value;
    document.getElementById('result-display').innerHTML = `You selected ${selectedOption}`;
});

document.getElementById('optimize-button').addEventListener('click', function() {
    // 여기에 서버로 데이터를 보내고 응답을 받는 AJAX 요청 구현
    alert('Optimization request sent!');
});
