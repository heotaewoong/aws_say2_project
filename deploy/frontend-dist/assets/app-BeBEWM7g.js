import{F as I}from"./browser-DaaRMyKQ.js";const o=n=>document.getElementById(n),s=(n,y="loading")=>{const i=o("status");i.textContent=n,i.className=`status ${y}`};I.oauth2.ready().then(async n=>{var y,i,v,$;s("✓ OAuth2 토큰 수신 완료. FHIR 서버에서 환자 정보 조회 중...","loading");try{const e=await n.patient.read();o("patient-section").style.display="block";const g=(y=e.name)==null?void 0:y[0],x=g?`${g.family} ${((i=g.given)==null?void 0:i.join(" "))||""}`:"(이름 없음)",T=e.birthDate?Math.floor((Date.now()-new Date(e.birthDate))/(365.25*24*60*60*1e3)):"?",w={male:"M",female:"F",other:"O"}[e.gender]||"?";o("patient-summary").innerHTML=`
      <div style="font-size: 18px; font-family: Georgia, serif; margin-bottom: 8px;">${x}</div>
      <div style="font-family: monospace; font-size: 13px; color: #64748B;">
        ${w} · ${T}세 · MRN: ${e.id} · DOB: ${e.birthDate||"N/A"}
      </div>
    `,o("patient-raw").textContent=JSON.stringify(e,null,2),s("✓ Patient OK. Conditions 조회 중...","loading");const B=await n.request(`Condition?patient=${e.id}&_count=20`);o("conditions-section").style.display="block";const u=((v=B.entry)==null?void 0:v.map(t=>t.resource))||[];o("conditions-list").innerHTML=u.length?u.map(t=>{var a,r,l,d,c,p,m,b,h,O;return`
          <div style="padding: 8px; border-bottom: 1px solid #E2E8F0; font-size: 13px;">
            <strong>${((a=t.code)==null?void 0:a.text)||((d=(l=(r=t.code)==null?void 0:r.coding)==null?void 0:l[0])==null?void 0:d.display)||"(name unknown)"}</strong>
            <span style="color: #64748B; font-family: monospace; font-size: 11px; margin-left: 8px;">
              ${((m=(p=(c=t.code)==null?void 0:c.coding)==null?void 0:p[0])==null?void 0:m.code)||""} · ${((O=(h=(b=t.clinicalStatus)==null?void 0:b.coding)==null?void 0:h[0])==null?void 0:O.code)||"unknown"}
            </span>
          </div>`}).join(""):'<div style="color: #64748B; font-size: 13px;">기록된 진단 없음</div>',s("✓ Conditions OK. Observations 조회 중...","loading");const D=await n.request(`Observation?patient=${e.id}&_count=10&_sort=-date`);o("observations-section").style.display="block";const f=(($=D.entry)==null?void 0:$.map(t=>t.resource))||[];o("observations-list").innerHTML=f.length?f.map(t=>{var a,r,l,d,c,p,m;return`
          <div style="padding: 8px; border-bottom: 1px solid #E2E8F0; font-size: 13px;">
            <strong>${((a=t.code)==null?void 0:a.text)||((d=(l=(r=t.code)==null?void 0:r.coding)==null?void 0:l[0])==null?void 0:d.display)||"(unknown)"}</strong>
            <span style="margin-left: 8px;">${((c=t.valueQuantity)==null?void 0:c.value)??""} ${((p=t.valueQuantity)==null?void 0:p.unit)||""}</span>
            <span style="color: #64748B; font-family: monospace; font-size: 11px; margin-left: 8px;">
              ${((m=t.effectiveDateTime)==null?void 0:m.split("T")[0])||""}
            </span>
          </div>`}).join(""):'<div style="color: #64748B; font-size: 13px;">기록된 검사 없음</div>',s(`✓ FHIR 연결 완료. 환자 ${x}, ${u.length} conditions, ${f.length} observations.`,"success"),o("next-step").style.display="block",sessionStorage.setItem("SMART_PATIENT_ID",e.id),sessionStorage.setItem("SMART_AUTHORIZED","true")}catch(e){console.error(e),s(`✗ 오류: ${e.message}`,"error")}}).catch(n=>{s(`✗ OAuth2 실패: ${n.message}`,"error"),console.error(n)});
//# sourceMappingURL=app-BeBEWM7g.js.map
