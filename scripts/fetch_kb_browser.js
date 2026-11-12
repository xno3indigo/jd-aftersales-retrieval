// 在浏览器中打开 https://help.jd.com/user/issue.html ，于开发者工具控制台运行本脚本，
// 会抓取下列分类下的全部条目正文，最后下载 kb.jsonl。
// 2026-10-23 抓取时，按下面的剔除规则和“正文不少于 20 字”的过滤，得到 129 篇。
// 帮助中心页面会更新，重新抓取的结果可能与项目中的 data/kb.jsonl 不完全一致。

const LISTS = {
  "售后政策（自营）": 112, "返修/退换货申请": 113, "服务单查询与修改": 114, "商品返回": 115, "退款说明": 116,
  "价格保护": 132, "大家电服务": 136, "售后到家": 139, "意外保护": 144,
  "发货与签收": 80, "京东配送服务": 81, "配送时效": 83, "配送异常": 84, "配送运费": 85,
  "订单修改": 49, "订单取消": 50, "订单拆分": 53, "订单异常": 54,
};
// 剔除：各地区逆向运费补偿明细表；正文为图片的安装收费页；与售后无关或仅适用于特定店铺的页面
const SKIP_IDS = new Set(["131-296", "131-297", "131-298", "131-1058", "131-1062",
  "103-983", "236-1470", "109-4026", "109-4539", "109-4565", "312-4554"]);

async function getDoc(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  let html = new TextDecoder("utf-8").decode(buf);
  const m = html.match(/charset=["']?([\w-]+)/i);
  if (m && !/utf-?8/i.test(m[1])) html = new TextDecoder(m[1]).decode(buf);
  return new DOMParser().parseFromString(html, "text/html");
}

function toText(el) {
  const c = el.cloneNode(true);
  c.querySelectorAll("script,style").forEach(x => x.remove());
  c.querySelectorAll("br").forEach(b => b.replaceWith("\n"));
  c.querySelectorAll("p,div,li,tr,h1,h2,h3,h4,h5,table").forEach(b => { b.prepend("\n"); b.append("\n"); });
  c.querySelectorAll("td,th").forEach(b => b.append(" | "));
  return c.textContent.replace(/ /g, " ").replace(/[ \t]+/g, " ").replace(/\n\s*\n+/g, "\n").trim();
}

(async () => {
  const seen = new Map();
  for (const [cat, id] of Object.entries(LISTS)) {
    const d = await getDoc(`/user/issue/list-${id}.html`);
    for (const a of d.querySelectorAll("a")) {
      const href = a.getAttribute("href") || "";
      const m = href.match(/\/user\/issue\/(\d+-\d+)\.html/);
      const title = a.textContent.replace(/^·\s*/, "").trim();
      if (!m || title.length < 3 || SKIP_IDS.has(m[1])) continue;
      const url = new URL(href, location.href).href;
      if (!seen.has(url)) seen.set(url, { id: m[1], cat, title, url });
    }
  }
  const lines = [];
  for (const v of seen.values()) {
    const cx = (await getDoc(v.url)).querySelector(".contxt");
    const text = cx ? toText(cx).replace(/\n\d{4}-\d{2}-\d{2}$/, "") : "";
    if (text.length >= 20) lines.push(JSON.stringify({ ...v, text }));
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "kb.jsonl" });
  a.click();
  console.log(`共 ${lines.length} 篇`);
})();
