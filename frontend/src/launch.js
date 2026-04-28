// SMART on FHIR launch endpoint.
// EHR이 ?iss=...&launch=... 로 호출하면 OAuth2 authorization 시작 → app.html로 redirect.
//
// 표준: SMART App Launch Framework v2.2.0
// 참조: https://hl7.org/fhir/smart-app-launch/
import FHIR from 'fhirclient';

FHIR.oauth2.authorize({
  // 우리 앱의 client_id (샌드박스에서는 어떤 값이든 등록만 되면 OK)
  client_id: 'rare-link-ai',

  // EHR-launched 시나리오에서 우리가 받을 권한 범위
  // launch: EHR launch context (선택된 환자 ID 자동 수신)
  // patient/*.read: 환자 데이터 읽기
  // openid fhirUser: 로그인한 의사 정보
  scope: 'launch patient/*.read openid fhirUser',

  // OAuth2 콜백 페이지 (token 받은 뒤 여기로 redirect)
  redirectUri: 'app.html',
});
