import { chromium } from 'playwright';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const imgDir = path.join(__dirname, 'images');

const BASE = 'https://d300v14l8u0wx7.cloudfront.net/?demo=1';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();

  console.log('1) Opening site...');
  await page.goto(BASE, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(3000);

  // Login
  await page.locator('button').filter({ hasText: /로그인/ }).first().click();
  await page.waitForTimeout(5000);

  // Click 열기 (it's an <a> tag)
  console.log('2) Clicking 열기...');
  await page.locator('a').filter({ hasText: /열기/ }).first().click();
  await page.waitForTimeout(3000);

  // Now click "EMR에서 환자 정보 불러오기"
  console.log('3) Clicking EMR 정보 불러오기...');
  const emrBtn = page.getByText('EMR 에서 환자 정보 불러오기', { exact: false });
  if (await emrBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
    await emrBtn.click();
    console.log('   Clicked EMR button, waiting for AI phases...');
    
    // Wait for each phase - total could be up to 30 seconds
    await page.waitForTimeout(5000);
    await page.screenshot({
      path: path.join(imgDir, 'screenshot_phase_progress.png'),
      fullPage: false,
    });
    console.log('   -> Captured phase in progress');
    
    await page.waitForTimeout(10000);
    await page.screenshot({
      path: path.join(imgDir, 'screenshot_phase_mid.png'),
      fullPage: false,
    });
    console.log('   -> Captured mid-phase');
    
    await page.waitForTimeout(15000);
    await page.screenshot({
      path: path.join(imgDir, 'screenshot_analysis_complete.png'),
      fullPage: false,
    });
    console.log('   -> Captured after phases');
    
    // Full page capture
    await page.screenshot({
      path: path.join(imgDir, 'screenshot_analysis_full.png'),
      fullPage: true,
    });
    console.log('   -> Captured full analysis page');

    // Try to find tabs like 감별진단, 소견서, etc. and click them
    const tabTexts = ['감별진단', '희귀질환', '소견서', 'CXR', 'X-ray', '히트맵'];
    for (const tabText of tabTexts) {
      const tab = page.getByText(tabText, { exact: false }).first();
      if (await tab.isVisible({ timeout: 2000 }).catch(() => false)) {
        console.log(`4) Clicking tab: ${tabText}`);
        await tab.click();
        await page.waitForTimeout(2000);
        await page.screenshot({
          path: path.join(imgDir, `screenshot_tab_${tabText}.png`),
          fullPage: false,
        });
      }
    }
    
  } else {
    console.log('   EMR button not found, trying alternative text...');
    // Try partial match
    const emrBtn2 = page.locator('button, a').filter({ hasText: /EMR|불러오기|정보/ }).first();
    if (await emrBtn2.isVisible({ timeout: 3000 }).catch(() => false)) {
      const txt = await emrBtn2.innerText();
      console.log(`   Found: ${txt}`);
      await emrBtn2.click();
      await page.waitForTimeout(30000);
      await page.screenshot({
        path: path.join(imgDir, 'screenshot_analysis_complete.png'),
        fullPage: false,
      });
    }
  }

  await browser.close();
  console.log('\n✅ Done!');
})();
