// background.js
//직접 사용되지는 않은 파일인데 혹시 몰라서 남겨놓을게요. 
chrome.runtime.onInstalled.addListener(() => {
    console.log("민원 분류기 확장 프로그램이 설치되었습니다.");
  });
  
  // 예: 특정 URL에 방문했을 때 알림을 주는 기능
  chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.url && changeInfo.url.includes('example.com')) {
      alert('example.com에 방문하셨습니다!');
    }
  });





  