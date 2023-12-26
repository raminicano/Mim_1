// 분류 시작 버튼 클릭 이벤트 리스너 추가
document.getElementById("submit").addEventListener("click", function() {
  // 선택한 지역, 민원 내용, 민원 제목 가져오기
  const region = document.getElementById("region").value;
  const complaint = document.getElementById("complaint").value;
  const title= document.getElementById("title").value;
  // API 엔드포인트 URL
  const apiUrl = "http://localhost:5000/api/analyze"; 

  // 요청 데이터 생성
  const requestData = {
    region: region,
    complaint: complaint,
    title: title
  };
 
  // 요청 옵션 설정
  const requestOptions = {
    method: "POST", // 데이터를 전송하기 위한 목적!
    headers: {
      "Content-Type": "application/json",
      "Accept": "*/*" // 새로이 추가한 파일. 
    },
    body: JSON.stringify(requestData)
  };

  // 서버에 POST 요청 보내기
  fetch(apiUrl, requestOptions)
    .then(response => response.json())
    .then(data => {
        const resultContainer = document.getElementById("result");
        if (data && typeof data === 'string') { // 데이터가 문자열인지 확인
            resultContainer.textContent = data; // JSON.stringify를 사용하지 않고 직접 값을 사용
        } else {
            resultContainer.textContent = "No data received from the server.";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        const resultContainer = document.getElementById("result");
        resultContainer.textContent = "An error occurred. Please try again.";
    });
});


