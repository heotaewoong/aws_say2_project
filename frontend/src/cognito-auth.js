/**
 * cognito-auth.js
 * Cognito REST API 직접 호출 — aws-amplify 의존성 없음
 * 계정: 666803869796 / 리전: ap-northeast-2
 */
import { COGNITO_CONFIG } from './aws-config.js';

const ENDPOINT = `https://cognito-idp.${COGNITO_CONFIG.region}.amazonaws.com/`;

function cognitoPost(target, body) {
  return fetch(ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': `AWSCognitoIdentityProviderService.${target}`,
    },
    body: JSON.stringify(body),
  });
}

/**
 * 이메일 + 비밀번호로 로그인
 * 성공: { IdToken, AccessToken, RefreshToken, ExpiresIn }
 * 임시 비밀번호 상태: Error('NEW_PASSWORD_REQUIRED')
 */
export async function cognitoSignIn(email, password) {
  const res = await cognitoPost('InitiateAuth', {
    AuthFlow: 'USER_PASSWORD_AUTH',
    ClientId: COGNITO_CONFIG.clientId,
    AuthParameters: {
      USERNAME: email,
      PASSWORD: password,
    },
  });

  const data = await res.json();

  if (data.ChallengeName === 'NEW_PASSWORD_REQUIRED') {
    throw new Error('NEW_PASSWORD_REQUIRED');
  }

  if (!res.ok || !data.AuthenticationResult) {
    // Cognito 에러 메시지 한국어 매핑
    const raw = data.__type || data.message || '로그인 실패';
    const msg = {
      'NotAuthorizedException': '이메일 또는 비밀번호가 올바르지 않습니다.',
      'UserNotFoundException': '등록되지 않은 계정입니다.',
      'UserNotConfirmedException': '이메일 인증이 완료되지 않았습니다.',
      'PasswordResetRequiredException': '비밀번호 재설정이 필요합니다.',
      'TooManyRequestsException': '요청이 너무 많습니다. 잠시 후 다시 시도하세요.',
    }[raw] || raw;
    throw new Error(msg);
  }

  return data.AuthenticationResult;
}

/**
 * 임시 비밀번호 → 새 비밀번호 변경 (첫 로그인 시)
 */
export async function cognitoRespondToNewPassword(email, tempPassword, newPassword) {
  // 1단계: 세션 토큰 획득
  const initRes = await cognitoPost('InitiateAuth', {
    AuthFlow: 'USER_PASSWORD_AUTH',
    ClientId: COGNITO_CONFIG.clientId,
    AuthParameters: { USERNAME: email, PASSWORD: tempPassword },
  });
  const initData = await initRes.json();
  const session = initData.Session;
  if (!session) throw new Error('세션 획득 실패');

  // 2단계: 새 비밀번호로 응답
  const res = await cognitoPost('RespondToAuthChallenge', {
    ChallengeName: 'NEW_PASSWORD_REQUIRED',
    ClientId: COGNITO_CONFIG.clientId,
    Session: session,
    ChallengeResponses: {
      USERNAME: email,
      NEW_PASSWORD: newPassword,
    },
  });
  const data = await res.json();
  if (!data.AuthenticationResult) throw new Error('비밀번호 변경 실패');
  return data.AuthenticationResult;
}

/**
 * 글로벌 로그아웃 (모든 기기에서 세션 무효화)
 */
export async function cognitoSignOut(accessToken) {
  if (!accessToken) return;
  await cognitoPost('GlobalSignOut', { AccessToken: accessToken });
}
