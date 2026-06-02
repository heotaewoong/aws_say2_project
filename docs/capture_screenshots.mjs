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
    deviceScaleFactor: 2,          // Retina quality
  });
  const page = await ctx.newPage();

  console.log('1) Opening site...');
  await page.goto(BASE, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(3000);  // wait for animations

  // Screenshot 1: Landing / Login page
  console.log('2) Capturing landing page...');
  await page.screenshot({
    path: path.join(imgDir, 'screenshot_landing.png'),
    fullPage: false,
  });

  // Try clicking login or any visible entry button
  const loginBtn = page.locator('button, [role="button"], a').filter({ hasText: /로그인|login|시작|enter|demo|launch/i }).first();
  if (await loginBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
    console.log('3) Clicking login/start button...');
    await loginBtn.click();
    await page.waitForTimeout(3000);
    await page.screenshot({
      path: path.join(imgDir, 'screenshot_after_login.png'),
      fullPage: false,
    });
  }

  // Try to find and click first patient in worklist
  const patientRow = page.locator('tr, [class*="patient"], [class*="row"], [class*="card"], [class*="item"], [class*="case"]')
    .filter({ hasText: /patient|환자|이름|P-|10/i }).first();
  if (await patientRow.isVisible({ timeout: 3000 }).catch(() => false)) {
    console.log('4) Clicking patient row...');
    await patientRow.click();
    await page.waitForTimeout(5000);  // wait for phase animations
    await page.screenshot({
      path: path.join(imgDir, 'screenshot_analysis.png'),
      fullPage: false,
    });
  }

  // Capture full page scrolled
  console.log('5) Capturing full page...');
  await page.screenshot({
    path: path.join(imgDir, 'screenshot_fullpage.png'),
    fullPage: true,
  });

  // Try to look for any other tabs or panels
  const tabs = page.locator('[role="tab"], [class*="tab"], button').filter({ hasText: /대시보드|dashboard|소견서|report|분석|analytics/i });
  const tabCount = await tabs.count();
  for (let i = 0; i < Math.min(tabCount, 3); i++) {
    const tab = tabs.nth(i);
    if (await tab.isVisible({ timeout: 2000 }).catch(() => false)) {
      const name = await tab.innerText().catch(() => `tab_${i}`);
      console.log(`6) Clicking tab: ${name}`);
      await tab.click();
      await page.waitForTimeout(2000);
      const safeName = name.replace(/[^a-zA-Z0-9가-힣]/g, '_').substring(0, 20);
      await page.screenshot({
        path: path.join(imgDir, `screenshot_tab_${safeName}.png`),
        fullPage: false,
      });
    }
  }

  // Get page title and all visible text for understanding the site structure
  const title = await page.title();
  const bodyText = await page.locator('body').innerText().catch(() => '');
  console.log('\n=== Page Title ===');
  console.log(title);
  console.log('\n=== Visible Text (first 2000 chars) ===');
  console.log(bodyText.substring(0, 2000));

  // List all clickable elements
  const buttons = await page.locator('button, [role="button"], a[href]').allInnerTexts();
  console.log('\n=== Clickable Elements ===');
  console.log(buttons.filter(t => t.trim()).join('\n'));

  await browser.close();
  console.log('\n✅ Done! Screenshots saved to docs/images/');
})();
