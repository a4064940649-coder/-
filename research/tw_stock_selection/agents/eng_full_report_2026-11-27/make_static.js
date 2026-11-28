const {chromium} = require('playwright');
const fs = require('fs');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({viewport: {width: 1100, height: 900}, deviceScaleFactor: 2});
  await p.route('**/*', r => r.request().url().startsWith('file://') ? r.continue() : r.abort());
  await p.goto('file://' + process.argv[2], {waitUntil: 'load'});
  await p.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 400) {
      window.scrollTo(0, y);
      await new Promise(r => setTimeout(r, 60));
    }
  });
  await p.waitForTimeout(2500);
  const n = await p.evaluate(() => document.querySelectorAll('.js-plotly-plot').length);
  console.log('plots', n);
  const els = await p.$$('.chart-container');
  for (let i = 0; i < els.length; i++) {
    const buf = await els[i].screenshot({type: 'jpeg', quality: 82});
    const d = 'data:image/jpeg;base64,' + buf.toString('base64');
    await els[i].evaluate((el, d) => {
      el.innerHTML = '<img src="' + d + '" style="width:100%;height:auto;display:block" alt="chart">';
    }, d);
  }
  // strip all script tags so the static copy has no JS
  await p.evaluate(() => {
    for (const s of Array.from(document.querySelectorAll('script'))) s.parentNode.removeChild(s);
  });
  const html = '<!DOCTYPE html>' + await p.evaluate(() => document.documentElement.outerHTML);
  fs.writeFileSync(process.argv[3], html);
  console.log('imgs', els.length, 'bytes', html.length);
  await b.close();
})();
