#!/usr/bin/env python3
"""Generate institutional equity-research-style PPT deck on global CSP CapEx."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from copy import deepcopy

# ----- Theme (券商研究報告風) -----
NAVY = RGBColor(0x0B, 0x1F, 0x3A)      # primary navy
GOLD = RGBColor(0xC8, 0xA0, 0x4A)      # accent gold
LIGHT = RGBColor(0xF4, 0xF6, 0xFA)
DARK_TEXT = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x6B, 0x72, 0x80)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x8C, 0x4A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT_TC = "Microsoft JhengHei"  # used in PPT once opened
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
blank = prs.slide_layouts[6]


def add_rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.5)
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, *, size=14, bold=False, color=DARK_TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT_TC):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    if isinstance(text, str):
        lines = text.split("\n")
    else:
        lines = text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_slide_header(slide, section_no, section_title, page_no, total_pages):
    # navy bar
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.7), NAVY)
    # gold accent
    add_rect(slide, 0, Inches(0.7), SLIDE_W, Inches(0.04), GOLD)
    add_text(slide, Inches(0.4), Inches(0.12), Inches(10), Inches(0.5),
             section_title, size=20, bold=True, color=WHITE)
    add_text(slide, Inches(11.4), Inches(0.18), Inches(1.6), Inches(0.4),
             f"Section {section_no}  |  {page_no}/{total_pages}",
             size=11, color=GOLD, align=PP_ALIGN.RIGHT)
    # footer
    add_rect(slide, 0, Inches(7.25), SLIDE_W, Inches(0.25), NAVY)
    add_text(slide, Inches(0.4), Inches(7.28), Inches(10), Inches(0.2),
             "全球 CSP 資本支出深度研究  ·  機構級科技產業研究  ·  2026/06",
             size=9, color=GOLD)
    add_text(slide, Inches(11.5), Inches(7.28), Inches(1.6), Inches(0.2),
             "Confidential | For Research Use", size=9, color=GOLD,
             align=PP_ALIGN.RIGHT)


def add_table(slide, x, y, w, h, data, *, header_fill=NAVY, header_color=WHITE,
              body_size=10, header_size=11, first_col_bold=False,
              col_widths=None, row_heights=None, zebra=True):
    rows = len(data); cols = len(data[0])
    tbl_shp = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = tbl_shp.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(w * cw / total))
    if row_heights:
        for i, rh in enumerate(row_heights):
            tbl.rows[i].height = Emu(int(rh))
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT_TC
            if r_idx == 0:
                r.font.size = Pt(header_size)
                r.font.bold = True
                r.font.color.rgb = header_color
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            else:
                r.font.size = Pt(body_size)
                r.font.color.rgb = DARK_TEXT
                if first_col_bold and c_idx == 0:
                    r.font.bold = True
                cell.fill.solid()
                if zebra and r_idx % 2 == 0:
                    cell.fill.fore_color.rgb = LIGHT
                else:
                    cell.fill.fore_color.rgb = WHITE
    return tbl


# ====================== COVER SLIDE ======================
s = prs.slides.add_slide(blank)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, NAVY)
# big diagonal gold strip
add_rect(s, 0, Inches(3.0), SLIDE_W, Inches(0.05), GOLD)
add_text(s, Inches(0.7), Inches(0.7), Inches(12), Inches(0.5),
         "INSTITUTIONAL EQUITY RESEARCH  ·  TECHNOLOGY", size=14, bold=True, color=GOLD)
add_text(s, Inches(0.7), Inches(1.4), Inches(12), Inches(1.2),
         "全球 CSP 資本支出 (CapEx)", size=44, bold=True, color=WHITE)
add_text(s, Inches(0.7), Inches(2.2), Inches(12), Inches(1.0),
         "深度研究與產業影響分析", size=32, bold=True, color=WHITE)
add_text(s, Inches(0.7), Inches(3.3), Inches(12), Inches(0.5),
         "Global Cloud Service Provider CapEx Deep-Dive  ·  2023 – 2026E",
         size=18, color=GOLD)
add_text(s, Inches(0.7), Inches(4.4), Inches(12), Inches(2.0),
         ["涵蓋：Amazon AWS  ·  Microsoft Azure  ·  Google Cloud  ·  Meta  ·  Oracle Cloud",
          "補充：xAI  ·  OpenAI/Stargate  ·  CoreWeave  ·  阿里雲  ·  騰訊雲",
          "",
          "▸ CapEx 全景  ▸ 暴增驅動力  ▸ 資金流向  ▸ AI 供應鏈地圖",
          "▸ 台股受惠族群  ▸ Bull / Base / Bear 情境  ▸ 面試加分洞察"],
         size=14, color=WHITE)
add_text(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.4),
         "Research Date: June 2026   |   Format: Sell-Side Research Note",
         size=11, color=GOLD)


# ====================== EXECUTIVE SUMMARY ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "EX", "① Executive Summary  關鍵結論", 2, 25)

# 6 insight cards in 2x3 grid
cards = [
    ("1.  史上最大 CapEx 週期",
     "四大美系雲端 CapEx：\n2023 ~$145B → 2025 ~$400B → 2026E ~$630–725B (+60~77% YoY)。\n五大合計 2026E 逼近 7,000 億美元。"),
    ("2.  本質是『供給回應已實現需求』",
     "推論已接棒訓練成主導工作負載；\nGoogle 月處理 token 達千兆級；\nOracle RPO 由 $98B → $638B (+363%)。\n『under-invest 風險 > over-invest』。"),
    ("3.  資金高度集中於『運算晶片』",
     "8卡 H100 伺服器 GPU 佔 BOM 約 83%；\nB200 BOM：HBM 45% / 邏輯 22% / CoWoS 17%。\n鐵三角：NVIDIA → 台積電(CoWoS) → SK海力士(HBM)。"),
    ("4.  瓶頸由晶片向實體外溢",
     "三大瓶頸：CoWoS、HBM、電力併網；\n變壓器 128 週交期、燃氣輪機 7 年；\n近半美國 2026 新建案延宕或取消。"),
    ("5.  財務代價已顯性化",
     "Alphabet 2026 FCF 估 -90%；\nAmazon FCF 估轉負；\nOracle FY26 FCF -$237 億；\nMeta CapEx/Revenue 約 39%。"),
    ("6.  台股是全球核心受惠地",
     "CoWoS 2330、3711、3443；\nODM 2317/2382/6669/3231；散熱 3324/3017；\n電源 2308；連接器 3533/3665；光通訊 3081/3105。"),
]
for i, (title, body) in enumerate(cards):
    row = i // 3; col = i % 3
    cx = Inches(0.4 + col * 4.25)
    cy = Inches(1.05 + row * 3.0)
    cw = Inches(4.05); ch = Inches(2.85)
    add_rect(s, cx, cy, cw, ch, LIGHT, line=NAVY)
    add_rect(s, cx, cy, Inches(0.12), ch, GOLD)
    add_text(s, cx + Inches(0.25), cy + Inches(0.15), cw - Inches(0.35), Inches(0.6),
             title, size=14, bold=True, color=NAVY)
    add_text(s, cx + Inches(0.25), cy + Inches(0.75), cw - Inches(0.35), ch - Inches(0.85),
             body, size=11, color=DARK_TEXT)


# ====================== ONE-LINE THESIS ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "EX", "① Executive Summary  ·  一句話投資結論", 3, 25)
add_rect(s, Inches(0.6), Inches(1.3), Inches(12.1), Inches(2.5), NAVY)
add_text(s, Inches(1.0), Inches(1.6), Inches(11.3), Inches(0.5),
         "INVESTMENT THESIS", size=13, bold=True, color=GOLD)
add_text(s, Inches(1.0), Inches(2.1), Inches(11.3), Inches(1.5),
         ["在『軍備競賽尚未見頂、瓶頸由晶片外溢至電力與封裝、",
          "  財務壓力開始浮現』的格局下——",
          "",
          "確定性最高的不是 AI 應用變現者，",
          "而是『不可替代的瓶頸環節』。"],
         size=18, bold=True, color=WHITE)
add_text(s, Inches(0.6), Inches(4.1), Inches(12.1), Inches(0.5),
         "▸  最值得超配的三大環節", size=15, bold=True, color=NAVY)

picks = [
    ("CoWoS / 先進製程", "台積電 2330、日月光投控 3711", "賣方市場、產能全數售罄至 2026"),
    ("HBM / 記憶體", "SK 海力士、Micron、Samsung", "B200 BOM 45%、Rubin 估佔 25%"),
    ("電力 / 散熱", "台達電 2308、雙鴻 3324、奇鋐 3017、Vertiv", "下一個外溢瓶頸、訂單能見度長"),
]
for i, (t1, t2, t3) in enumerate(picks):
    cy = Inches(4.7 + i * 0.78)
    add_rect(s, Inches(0.6), cy, Inches(0.15), Inches(0.65), GOLD)
    add_text(s, Inches(0.9), cy + Inches(0.05), Inches(3.0), Inches(0.6),
             t1, size=14, bold=True, color=NAVY)
    add_text(s, Inches(4.2), cy + Inches(0.05), Inches(4.2), Inches(0.6),
             t2, size=12, color=DARK_TEXT)
    add_text(s, Inches(8.6), cy + Inches(0.05), Inches(4.2), Inches(0.6),
             t3, size=12, color=GREY)


# ====================== SECTION 1: CAPEX OVERVIEW TABLE ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "1", "① CSP CapEx 全景分析 — 近三年絕對值 ＋ 2026 指引 (USD bn)", 4, 25)
data = [
    ["公司 / 口徑", "2023", "2024", "2025", "2026E 指引", "3年 CAGR (23→26E)", "主來源"],
    ["Amazon (PP&E)", "48.1", "77.7", "~118–125", "~200", "~60%", "10-K / Q4 法說會"],
    ["Microsoft (含融資租賃)", "~28", "~55.7", "~88", ">120 (媒體 ~185–190)", "~90%", "8-K / FY 法說會"],
    ["Alphabet (PP&E)", "32.3", "52.5", "~91", "180–190", "~80%", "10-K / 8-K"],
    ["Meta (PP&E)", "28.1", "39.2", "~70–72", "125–145", "~75%", "10-K / 8-K"],
    ["四大合計", "~145", "~225", "~388–410", "~630–725", "~67%", "CNBC / Tom's HW"],
    ["Oracle (FY, 5月底)", "7.0 (FY24)", "21.2 (FY25)", "~35+ (FY26)", "FY27 大幅躍升", ">70%", "8-K / 法說會"],
]
add_table(s, Inches(0.4), Inches(1.05), Inches(12.5), Inches(3.7), data,
          col_widths=[2.4, 1.0, 1.0, 1.3, 1.9, 1.5, 2.0], first_col_bold=True,
          body_size=12, header_size=12)

add_rect(s, Inches(0.4), Inches(5.0), Inches(12.5), Inches(2.1), LIGHT, line=NAVY)
add_text(s, Inches(0.6), Inches(5.15), Inches(12), Inches(0.4),
         "▸  研究員洞察", size=13, bold=True, color=NAVY)
add_text(s, Inches(0.6), Inches(5.55), Inches(12.2), Inches(1.5),
         ["• 2024 YoY +55% → 2025 +73% → 2026E +60~77%，加速度仍未轉折；五大公司合計 2022→2025 三年 ×2.76 倍。",
          "• Microsoft 投資強度最健康（Azure+M365 雙引擎現金流對沖）；Meta、Oracle 最激進（無對外雲端對沖／槓桿擴張），財務尾部風險最高。",
          "• Oracle RPO：$98B (FY24) → $523B (Q2 FY25) → $638B (FY26, +363%)，其中 Q3/Q4 大型 AI 合約預付/客戶自帶 GPU 達 $750 億——這是『需求真實』最有力證據。"],
         size=11, color=DARK_TEXT)


# ====================== SLIDE: FCF & CAPEX INTENSITY ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "1", "① CSP CapEx 全景 — 財務體質與投資強度 (2025E)", 5, 25)
data = [
    ["公司", "2025 CapEx", "2026 FCF 預估", "CapEx/Revenue", "CapEx/OCF", "財務訊號"],
    ["Amazon", "~$120B", "轉負 -$17~-28B (MS/BofA)", "~17%", "~95–100%+", "零售現金流被 AWS CapEx 吃光"],
    ["Microsoft", "~$88B", "仍為正但收斂", "~31%", "~65–70%", "體質最穩、利潤率最高"],
    ["Alphabet", "~$91B", "$73B → ~$8B (-90%, Pivotal)", "~23%", "~60%", "FCF 將急縮"],
    ["Meta", "~$70–72B", "收斂中", "~39%", "~75%", "投資強度最高、無雲端營收對沖"],
    ["Oracle", "~$21B (FY25)", "FY26: -$23.7B", "~37%", "轉負", "槓桿擴張、賭 OCI 放量"],
]
add_table(s, Inches(0.4), Inches(1.05), Inches(12.5), Inches(3.4), data,
          col_widths=[1.4, 1.4, 2.6, 1.6, 1.5, 4.0], first_col_bold=True,
          body_size=11, header_size=12)

add_rect(s, Inches(0.4), Inches(4.7), Inches(6.0), Inches(2.4), LIGHT, line=NAVY)
add_text(s, Inches(0.55), Inches(4.82), Inches(5.8), Inches(0.4),
         "▸  新雲與中國雲補充", size=13, bold=True, color=NAVY)
add_text(s, Inches(0.55), Inches(5.2), Inches(5.8), Inches(1.9),
         ["• OpenAI / Stargate：$500B 四年計畫",
          "• xAI Colossus：田納西 Memphis、GW 級電力",
          "• CoreWeave：GPU 抵押、高槓桿擴張",
          "• 阿里雲：三年 3,800 億 RMB (~$530 億)",
          "• 騰訊雲：受美高階 GPU 出口管制制約"],
         size=11, color=DARK_TEXT)

add_rect(s, Inches(6.6), Inches(4.7), Inches(6.3), Inches(2.4), NAVY)
add_text(s, Inches(6.75), Inches(4.82), Inches(6.0), Inches(0.4),
         "▸  關鍵洞察", size=13, bold=True, color=GOLD)
add_text(s, Inches(6.75), Inches(5.2), Inches(6.0), Inches(1.9),
         ["『現金流斷頭』是 2026 年市場最大爭點：",
          "Alphabet FCF 估降 90%、Amazon 轉負、",
          "Oracle 大幅負值——但 Microsoft 仍能",
          "用雙引擎現金流支撐 CapEx，",
          "這是它估值溢價的核心理由。"],
         size=12, color=WHITE)


# ====================== SECTION 2 INTRO ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "2", "② CapEx 為何暴增？ — 需求側『五層放大器』", 6, 25)

layers = [
    ("①  AI 模型規模成長",
     "GPT/Gemini/Claude/Llama 每代訓練算力 ×4–10；叢集由萬卡走向 10 萬卡、GW 級。"),
    ("②  Token 需求暴增",
     "Google 月處理 token 達千兆 (quadrillion) 級並持續倍增——token ≒ 雲端 CapEx 計費單位。"),
    ("③  推論 (Inference) 需求爆發",
     "Reasoning model 單次查詢 token ×10–100；推論已成為主導工作負載——可即時變現。"),
    ("④  Agentic AI",
     "Agent = 規劃→工具→反思→再呼叫；單一任務觸發數十至數百次推論，再放大一個量級。"),
    ("⑤  AI 搜尋與 AI 辦公",
     "AI Overviews / Copilot / Gemini 嵌入每次搜尋、郵件、文件——把實驗轉為全民日常負載。"),
]
for i, (h, b) in enumerate(layers):
    y = Inches(1.1 + i * 1.18)
    add_rect(s, Inches(0.4), y, Inches(0.5), Inches(1.0), NAVY)
    add_text(s, Inches(0.4), y + Inches(0.3), Inches(0.5), Inches(0.4),
             str(i+1), size=22, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(0.95), y, Inches(11.95), Inches(1.0), LIGHT, line=NAVY)
    add_text(s, Inches(1.15), y + Inches(0.1), Inches(11.5), Inches(0.4),
             h, size=14, bold=True, color=NAVY)
    add_text(s, Inches(1.15), y + Inches(0.5), Inches(11.5), Inches(0.5),
             b, size=11, color=DARK_TEXT)


# ====================== SECTION 2: ROI LOGIC ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "2", "② 為何雲端巨頭願意投入數百億？ — ROI 邏輯推導", 7, 25)
data = [
    ["推導角度", "邏輯", "證據"],
    ["1. 機會成本 > 投資成本",
     "平台轉移期，輸掉容量＝輸掉下一個十年雲端市佔；『under-invest 風險 > over-invest』",
     "Nadella / Pichai / Zuckerberg 法說會一致口徑"],
    ["2. 需求受供給約束",
     "雲端業者反覆表示：『我們被容量限制，不是被需求限制』——蓋多少賣多少",
     "MS / Google / Oracle 多次法說會；Oracle RPO $638B"],
    ["3. 基礎設施可重配置",
     "GPU 與資料中心今天訓練、明天推論、後天內部產品——資產用途可轉換，降低押錯風險",
     "Amazon / Meta 管理層論述"],
    ["4. 變現路徑已出現",
     "Azure AI 為 Azure 成長主力；AWS / GCP backlog 創高；Oracle OCI 預定 FY29 達 $1,440 億",
     "各雲端營收與 backlog / RPO"],
    ["5. 折舊攤提 ＋ 雲端毛利",
     "伺服器 5–6 年折舊；搭配高毛利雲端服務，理論 IRR 為正——前提是利用率維持高檔",
     "各公司折舊政策（Bear 爭點：見第 6 部分）"],
]
add_table(s, Inches(0.4), Inches(1.05), Inches(12.5), Inches(4.8), data,
          col_widths=[2.4, 6.1, 4.0], first_col_bold=True,
          body_size=11, header_size=12)
add_rect(s, Inches(0.4), Inches(6.1), Inches(12.5), Inches(1.0), NAVY)
add_text(s, Inches(0.6), Inches(6.25), Inches(12), Inches(0.7),
         "▸  本質洞察：這是一場『容量軍備競賽下的囚徒困境』——只要對手在蓋，沒人敢停。\n     理性個體決策（不能落後）匯總成總體過度投資傾向，既是 Bull 也是 Bear 的根源。",
         size=13, bold=True, color=WHITE)


# ====================== SECTION 3: CAPEX FLOW ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "3", "③ CapEx 投資流向拆解 — 一美元 AI CapEx 流向哪裡？", 8, 25)

# left: framework
add_rect(s, Inches(0.4), Inches(1.05), Inches(5.5), Inches(6.0), NAVY)
add_text(s, Inches(0.6), Inches(1.2), Inches(5.2), Inches(0.4),
         "1 GW AI 資料中心 ≈ $380 億", size=16, bold=True, color=GOLD)
add_text(s, Inches(0.6), Inches(1.65), Inches(5.2), Inches(0.4),
         "（Epoch AI 模型）", size=11, color=WHITE)
add_text(s, Inches(0.6), Inches(2.2), Inches(5.2), Inches(0.5),
         "IT 設備 (伺服器+網路)", size=14, bold=True, color=WHITE)
add_rect(s, Inches(0.6), Inches(2.65), Inches(5.0), Inches(0.4), GOLD)
add_text(s, Inches(0.7), Inches(2.7), Inches(4.8), Inches(0.35),
         "~60–63%", size=14, bold=True, color=NAVY)
add_text(s, Inches(0.6), Inches(3.25), Inches(5.2), Inches(0.5),
         "廠房 / 電力 / 散熱 / 土地", size=14, bold=True, color=WHITE)
add_rect(s, Inches(0.6), Inches(3.7), Inches(3.2), Inches(0.4), GREY)
add_text(s, Inches(0.7), Inches(3.75), Inches(3.0), Inches(0.35),
         "~35–40%", size=14, bold=True, color=WHITE)
add_text(s, Inches(0.6), Inches(4.4), Inches(5.2), Inches(2.6),
         ["參考錨點：",
          "• AI 伺服器 2024 $187B (+69%, 佔 65%)",
          "• AI 伺服器 2025 $298B → 2026 佔 74%",
          "• IDC：2025 伺服器市場 $444B (+80%)",
          "• GPU 加速伺服器已佔逾半"],
         size=11, color=WHITE)

# right: BOM table
data = [
    ["項目", "估算占比", "成長", "瓶頸 / 來源"],
    ["AI 伺服器 (整機)", "最大單一項", "極高", "整合產能 / CoWoS"],
    ["└ GPU/加速器", "AI 伺服器 BOM 70–83%", "極高", "晶片產能 (SemiAnalysis)"],
    ["└ HBM", "B200 BOM 45% (~$2,900/$6,400)", "爆發", "售罄至 2026 (TrendForce)"],
    ["└ 邏輯晶粒 / CoWoS-L", "B200: 22% / 17%", "高", "CoWoS 售罄 (TrendForce)"],
    ["└ CPU", "個位數 %", "平穩", "—"],
    ["└ NAND / eSSD", "~14.7% (CPU server)", "高", "+70% QoQ (TrendForce)"],
    ["網路交換器", "~10–15%", "高", "800G/1.6T 切換 (Dell'Oro)"],
    ["光模組", "隨 GPU 線性成長", "2.6x", "EML 雷射 (LightCounting)"],
    ["液冷系統", "隨 GB200/300 滲透", "爆發", "CDU/快接頭 (Dell'Oro)"],
    ["電力設備", "廠房側大項", "高", "電網併網/變壓器 (IEA)"],
    ["資料中心建置/土地", "占 35–40% 之主體", "高", "土地 / 電力配額"],
]
add_table(s, Inches(6.1), Inches(1.05), Inches(6.8), Inches(6.0), data,
          col_widths=[2.0, 2.0, 0.9, 2.0], first_col_bold=True,
          body_size=10, header_size=11)


# ====================== SECTION 3: KEY BOTTLENECKS ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "3", "③ 關鍵零組件市場數據 — 成長速度與瓶頸", 9, 25)
data = [
    ["環節", "市場規模", "成長速度", "瓶頸 / 關鍵數據", "受惠者"],
    ["HBM",
     "$18.2B (2024) → $46.7B (2025)",
     "+156% YoY",
     "SK 海力士、Micron 2026 產能全數售罄；HBM 佔 DRAM 產值 ~33%",
     "SK Hynix / Micron / Samsung"],
    ["NAND / eSSD",
     "4Q25 eSSD +28% QoQ (~$6.54B)",
     "1Q26 +53–58%、2Q26 +70–75% QoQ",
     "QLC 122TB/245TB；AI 推論需求結構性轉變",
     "Solidigm / SanDisk / Kioxia"],
    ["CoWoS",
     "2024 ~3.5–4 萬片/月 → 2025 ~7.5 萬 → 2026 目標 12–13 萬",
     "CAGR >50%",
     "NVIDIA 預訂 >60%；CoWoS-L/S 已售罄；交期 50+ 週",
     "台積電 2330 / 日月光 3711 / Amkor / 創意 3443"],
    ["光模組",
     "整體市場 >$230 億 (2025, +50%); 800G+ 由 2,400 萬 → 6,300 萬顆 (2026)",
     "+2.6x YoY",
     "200G/lane EML 雷射緊缺；NVIDIA 砸 $40 億鎖定 Lumentum / Coherent",
     "InnoLight / Coherent / Lumentum / Fabrinet; 聯亞 3081"],
    ["網路",
     "DC 交換器 2025Q3 創 $80 億／季",
     "AI 後端以太網佔比 <50% → 2/3",
     "NVIDIA 網路單季 $81.9 億 (+162%)；Broadcom AI FY26 ~$56B",
     "NVIDIA / Arista / Cisco / Broadcom"],
    ["液冷",
     "$30 億 (2025) → $70 億 (2029)",
     "CAGR ~20%",
     "GB200/300 標配；單櫃液冷 ~$50K",
     "Vertiv / 雙鴻 3324 / 奇鋐 3017 / 台達 2308"],
    ["電力",
     "Vertiv 在手單 $150 億 (+109%); Eaton DC 在手 228 GW",
     "資料中心用電 2025 +17%",
     "變壓器 128 週、燃氣輪機 7 年；近半 2026 美國新建案延宕",
     "Vertiv / Eaton / GE Vernova / 台達 2308"],
]
add_table(s, Inches(0.3), Inches(1.05), Inches(12.7), Inches(6.0), data,
          col_widths=[1.4, 2.5, 1.6, 3.6, 3.6], first_col_bold=True,
          body_size=10, header_size=11)


# ====================== SECTION 3: RACK BOM EVOLUTION ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "3", "③ 機櫃級成本經濟學 — GB200 → GB300 → Vera Rubin", 10, 25)

# Bar chart visualization
ys = [Inches(2.5), Inches(2.5), Inches(2.5)]
xs = [Inches(1.5), Inches(5.3), Inches(9.1)]
labels = ["GB200 NVL72\n~$3.0M", "GB300 NVL72\n~$6.0–6.5M", "Vera Rubin VR200 NVL72\n~$7.8M"]
heights = [Inches(1.8), Inches(3.2), Inches(4.0)]
colors = [NAVY, NAVY, GOLD]

for i in range(3):
    add_rect(s, xs[i], Inches(6.0) - heights[i], Inches(2.8), heights[i], colors[i])
    add_text(s, xs[i], Inches(6.05), Inches(2.8), Inches(0.8),
             labels[i], size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

add_text(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(0.5),
         "▸  每一代 NVLink 機櫃的記憶體含量正快速吃掉成本結構", size=15, bold=True, color=NAVY)
add_text(s, Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.5),
         "Vera Rubin：HBM4+LPDDR5X 達 $2M (佔 25%, vs GB300 +485%)；NAND 由 ~0 → $1M+",
         size=12, color=GREY)

add_rect(s, Inches(0.4), Inches(6.9), Inches(12.5), Inches(0.3), NAVY)
add_text(s, Inches(0.6), Inches(6.94), Inches(12), Inches(0.25),
         "洞察：HBM／儲存供應鏈是『含量倍增』結構性順風；ODM 自 Vera Rubin 起恐被收斂至 L10，毛利承壓。",
         size=11, bold=True, color=GOLD)


# ====================== SECTION 4: SUPPLY CHAIN MAP ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "4", "④ AI Infrastructure 供應鏈地圖", 11, 25)

chain = [
    ("CSP", "Amazon · Microsoft · Google · Meta · Oracle · OpenAI · xAI · 阿里 · 騰訊"),
    ("資料中心", "土地 / 廠房 / 電力 / 併網 — Vertiv / Eaton / Schneider / GE Vernova"),
    ("伺服器 / 機櫃", "鴻海 2317 · 廣達 2382 · 緯穎 6669 · 緯創 3231 · 英業達 2356 · Supermicro · Dell"),
    ("GPU / 加速器", "NVIDIA · AMD · Broadcom (ASIC) · Marvell · TPU / Trainium / Maia / MTIA"),
    ("製造 / 封裝", "台積電 2330 · 日月光 3711 · 創意 3443 · Amkor · Intel 18A"),
    ("記憶體 (HBM)", "SK Hynix · Samsung · Micron"),
    ("網通 / 光通訊", "Arista · Cisco · Broadcom · NVIDIA / InnoLight · Coherent · Lumentum / 3081 · 3105"),
    ("散熱 / 電力", "Vertiv · Boyd / 台達 2308 · 雙鴻 3324 · 奇鋐 3017 · 光寶 2301"),
]
for i, (layer, names) in enumerate(chain):
    y = Inches(1.05 + i * 0.75)
    add_rect(s, Inches(0.4), y, Inches(2.6), Inches(0.65), NAVY)
    add_text(s, Inches(0.5), y + Inches(0.15), Inches(2.5), Inches(0.4),
             layer, size=13, bold=True, color=WHITE)
    add_rect(s, Inches(3.0), y, Inches(9.9), Inches(0.65), LIGHT, line=NAVY)
    add_text(s, Inches(3.15), y + Inches(0.15), Inches(9.7), Inches(0.4),
             names, size=11, color=DARK_TEXT)
    if i < len(chain) - 1:
        # arrow indicator
        tri = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(1.6), y + Inches(0.65), Inches(0.3), Inches(0.1))
        tri.fill.solid(); tri.fill.fore_color.rgb = GOLD
        tri.line.fill.background()


# ====================== SECTION 4: BENEFICIARY TIER ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "4", "④ 全球與台灣主要供應商 — 受惠層級分類", 12, 25)
data = [
    ["供應鏈層", "全球供應商", "台灣供應商 (代號)", "受惠層級"],
    ["GPU / 加速器", "NVIDIA、AMD", "創意 3443 (IP)", "核心"],
    ["客製 ASIC", "Broadcom、Marvell (~95% 協同設計)", "世芯-KY 3661、創意 3443、聯發科 2454", "核心"],
    ["晶圓 / 先進封裝", "TSMC、Intel 18A、Amkor", "台積電 2330、日月光 3711、創意 3443", "核心 (CoWoS 瓶頸)"],
    ["HBM / 記憶體", "SK Hynix、Samsung、Micron", "(後段測試) —", "核心"],
    ["伺服器 ODM/機櫃", "Supermicro、Dell、HPE", "鴻海 2317、廣達 2382、緯穎 6669、緯創 3231", "核心"],
    ["網路交換器", "Arista、Cisco、Broadcom、NVIDIA", "智邦 2345、智易 3596", "核心 / 次"],
    ["光通訊 (模組)", "InnoLight、Eoptolink、Coherent、Lumentum、Fabrinet", "聯亞 3081、穩懋 3105、聯鈞 4979、波若威 3163", "核心(上游)/次"],
    ["連接器 / 銅纜", "Amphenol、TE Connectivity", "嘉澤 3533、貿聯 3665、健和興", "核心"],
    ["散熱 (液冷)", "Vertiv、Boyd、CoolIT", "雙鴻 3324、奇鋐 3017、台達 2308", "核心"],
    ["電源", "Vertiv、Eaton、Schneider", "台達電 2308、光寶 2301", "核心"],
    ["PCB / CCL / 載板", "Ibiden、Shinko", "欣興 3037、台燿 6274、台光電 2383、金像電 2368", "核心 / 次"],
]
add_table(s, Inches(0.3), Inches(1.05), Inches(12.7), Inches(6.05), data,
          col_widths=[1.7, 3.4, 5.0, 1.4], first_col_bold=True,
          body_size=10, header_size=11)


# ====================== SECTION 5: TAIWAN BENEFICIARIES ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "5", "⑤ 台股受惠族群分析 — 動能與風險", 13, 25)
data = [
    ["族群", "代表公司 (代號)", "關聯度", "三年成長動能", "最大風險"],
    ["伺服器 ODM", "鴻海 2317、廣達 2382、緯穎 6669、緯創 3231", "極高", "AI 伺服器 ASP 數十倍於傳統機", "毛利偏低、CSP 議價、規格急變"],
    ["網通", "智邦 2345、智易 3596", "高", "800G→1.6T 升級", "白牌競爭、以太網 vs IB 路線"],
    ["光通訊", "聯亞 3081、穩懋 3105、聯鈞 4979、波若威 3163", "中高 (上游最稀缺)", "2.6x 出貨、EML 缺貨、CPO", "模組組裝段被中廠壟斷"],
    ["散熱 (液冷)", "雙鴻 3324、奇鋐 3017", "極高", "氣冷轉液冷滲透率跳升", "技術驗證、價格競爭"],
    ["PCB / CCL", "欣興 3037、台燿 6274、台光電 2383、金像電 2368", "高", "高速 M8/M9 材料升級", "原料、認證週期"],
    ["電源", "台達電 2308、光寶 2301", "極高", "GW 級資料中心、HVDC", "競爭、原物料"],
    ["連接器 / 銅纜", "嘉澤 3533、貿聯 3665", "高", "貿聯 GB200 ~35% 份額、單櫃 $12–14K", "規格快速演進、轉光損失"],
    ["機殼 / 滑軌", "勤誠 8210、川湖 8059", "中高", "機櫃量增", "規模小、題材集中"],
    ["CoWoS / 封測", "台積電 2330、日月光 3711、創意 3443", "核心瓶頸", "CoWoS 產能近翻倍", "客戶過度集中於 NVIDIA"],
]
add_table(s, Inches(0.3), Inches(1.05), Inches(12.7), Inches(5.8), data,
          col_widths=[1.6, 3.6, 1.3, 3.4, 2.8], first_col_bold=True,
          body_size=10, header_size=11)

add_rect(s, Inches(0.3), Inches(6.95), Inches(12.7), Inches(0.25), NAVY)
add_text(s, Inches(0.45), Inches(6.98), Inches(12.5), Inches(0.22),
         "▸  族群梯隊：第一(確定+不可替代) CoWoS+電源+散熱  ▸  第二 ODM+連接器  ▸  第三(高彈性) 光通訊上游+高速 CCL",
         size=10, bold=True, color=GOLD)


# ====================== SECTION 5: ODM SHARE BAR ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "5", "⑤ 量化卡位 — GB200/GB300 NVL72 ODM 機櫃市佔 (2025)", 14, 25)

shares = [("Foxconn / FII (2317)", 52, GOLD),
          ("Wistron (3231)", 21, NAVY),
          ("Quanta (2382)", 19, NAVY),
          ("Wiwynn (6669) + Others", 8, GREY)]
bar_x = Inches(0.6); bar_y = Inches(2.2); bar_w = Inches(8.0); bar_h = Inches(0.55)
for i, (name, pct, color) in enumerate(shares):
    y = bar_y + Inches(i * 0.95)
    add_text(s, Inches(0.4), y, Inches(2.6), Inches(0.5),
             name, size=12, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, Inches(3.1), y + Inches(0.08), bar_w, bar_h, LIGHT)
    add_rect(s, Inches(3.1), y + Inches(0.08), Inches(bar_w.inches * pct / 60), bar_h, color)
    add_text(s, Inches(11.2), y, Inches(1.5), Inches(0.5),
             f"{pct}%", size=14, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)

add_text(s, Inches(0.4), Inches(1.1), Inches(12), Inches(0.5),
         "▸  2025 全年 NVL72 ~2.7–2.9 萬台 → 2026 目標 7–8 萬台 (~2x)；鴻海份額看升至 55–60%",
         size=14, bold=True, color=NAVY)

# Right side context
add_rect(s, Inches(0.4), Inches(6.0), Inches(12.5), Inches(1.2), NAVY)
add_text(s, Inches(0.6), Inches(6.1), Inches(12), Inches(0.4),
         "▸  CSP 主要對應", size=13, bold=True, color=GOLD)
add_text(s, Inches(0.6), Inches(6.5), Inches(12), Inches(0.7),
         "Foxconn ↔ Microsoft(GB200 最大買家)/Oracle/AWS/Google/CoreWeave   |   Quanta ↔ Meta/AWS/Google/Oracle",
         size=11, color=WHITE)
add_text(s, Inches(0.6), Inches(6.75), Inches(12), Inches(0.7),
         "Wiwynn ↔ Meta(主)/Microsoft   |   Wistron ↔ Microsoft/AWS/Meta   |   Dell/SMCI ↔ CoreWeave/xAI",
         size=11, color=WHITE)
# Sources: TrendForce/36kr; Morgan Stanley; DigiTimes
add_text(s, Inches(0.4), Inches(7.05), Inches(8), Inches(0.2),
         "Source: TrendForce/36kr、Morgan Stanley、DigiTimes、Ming-Chi Kuo",
         size=9, color=GREY)


# ====================== SECTION 6: KEY DEBATES ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "6", "⑥ 市場最關注的議題 — 六大爭點速覽", 15, 25)
debates = [
    ("1.  AI CapEx 是否過熱？",
     "2026 ~$7,000 億 (+60~77%)。Goldman: 2025–27 累計 $1.15T；Dell'Oro: 2030 DC capex $1.7T；McKinsey: 2030 累計 $6.7T"),
    ("2.  ROI 是否下降？",
     "Sequoia (David Cahn) $6,000 億營收缺口；Goldman (Jim Covello) 『$1T capex 換來什麼』vs 樂觀派"),
    ("3.  AI 泡沫風險？",
     "折舊年限拉長至 5–6 年；NVIDIA↔OpenAI↔Oracle↔CoreWeave 循環融資；Burry/Chanos 質疑獲利被美化"),
    ("4.  電力供應問題",
     "IEA DC 用電 2030 翻倍至 945 TWh；變壓器 128 週、燃氣輪機 7 年；近半美國 2026 新建案延宕"),
    ("5.  GPU 供應是否過剩？",
     "Blackwell/HBM 售罄 vs 舊 H100 租金鬆動；GPU 折舊懸崖 (depreciation cliff) 疑慮"),
    ("6.  推論是否接棒訓練？",
     "若推論 (可變現) 確為主體，ROI 論點成立——化解泡沫疑慮的關鍵"),
]
for i, (h, b) in enumerate(debates):
    row = i // 2; col = i % 2
    cx = Inches(0.4 + col * 6.35)
    cy = Inches(1.1 + row * 2.0)
    add_rect(s, cx, cy, Inches(6.2), Inches(1.85), LIGHT, line=NAVY)
    add_rect(s, cx, cy, Inches(0.12), Inches(1.85), GOLD)
    add_text(s, cx + Inches(0.25), cy + Inches(0.1), Inches(5.9), Inches(0.5),
             h, size=14, bold=True, color=NAVY)
    add_text(s, cx + Inches(0.25), cy + Inches(0.55), Inches(5.9), Inches(1.25),
             b, size=11, color=DARK_TEXT)


# ====================== SECTION 6: BULL / BASE / BEAR ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "6", "⑥ Bull / Base / Bear 三情境分析", 16, 25)

cols = [
    ("🐂  Bull Case", "~35%", GREEN,
     ["需求受供給約束、推論變現中、平台轉移不容缺席",
      "2026 >$7,000 億、2027 續創高",
      "Oracle RPO $638B、Azure AI 放量、token 千兆級",
      "瓶頸 = 定價權 (賣方市場延續)",
      "歷史類比：雲端 / 行動網路初期超前投資",
      "受惠：全產業鏈普漲，瓶頸環節最甜"]),
    ("⚖️  Base Case", "~45%", NAVY,
     ["CapEx 續增但增速放緩，ROI 拉長但為正",
      "2026 達標；2027 增速由 +70% → +20~30%",
      "backlog 真實，但變現有時間差",
      "瓶頸逐步緩解、ASP 回落",
      "歷史類比：鐵路 / 電力等長週期回報",
      "受惠：分化——瓶頸環節 > 同質代工"]),
    ("🐻  Bear Case", "~20%", RED,
     ["營收追不上 CapEx，折舊與電力反噬獲利",
      "2027 出現『數位化消化期』、個別玩家砍單",
      "FCF 崩跌：GOOG -90%、AMZN 轉負、ORCL -$237 億",
      "電力 / 折舊成為成本黑洞",
      "歷史類比：1999–2001 電信光纖過度建設",
      "受傷：高槓桿 neocloud、低毛利 ODM、過剩 GPU"]),
]
for i, (h, p, color, items) in enumerate(cols):
    x = Inches(0.3 + i * 4.4)
    add_rect(s, x, Inches(1.05), Inches(4.2), Inches(0.7), color)
    add_text(s, x, Inches(1.15), Inches(4.2), Inches(0.5),
             h, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, x, Inches(1.45), Inches(4.2), Inches(0.3),
             f"主觀機率 {p}", size=11, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s, x, Inches(1.75), Inches(4.2), Inches(5.0), LIGHT, line=color)
    for j, it in enumerate(items):
        add_text(s, x + Inches(0.2), Inches(1.95 + j * 0.78), Inches(3.8), Inches(0.75),
                 "• " + it, size=10, color=DARK_TEXT)


# ====================== BASE CASE TAKEAWAY ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "6", "⑥ 本報告基準觀點 (Base Case)", 17, 25)
add_rect(s, Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.5), NAVY)
add_text(s, Inches(1.0), Inches(1.8), Inches(11.3), Inches(0.5),
         "BASE CASE — OUR HOUSE VIEW", size=14, bold=True, color=GOLD)
add_text(s, Inches(1.0), Inches(2.4), Inches(11.3), Inches(1.5),
         ["需求是真實的——",
          "Oracle RPO 與 token 數據難以造假；",
          "但『需求真實』≠ 每一塊錢 CapEx 都有合理回報。"],
         size=18, bold=True, color=WHITE)

add_text(s, Inches(1.0), Inches(4.5), Inches(11.3), Inches(0.5),
         "2026–2027 最可能劇本：", size=15, bold=True, color=GOLD)
add_text(s, Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.8),
         ["『總量續創高、內部分化加劇』",
          "",
          "▸  擁有定價權的瓶頸環節（CoWoS、HBM、電力、散熱）持續受惠",
          "▸  同質化、無對沖現金流、依賴外部融資的玩家將首先承壓"],
         size=14, color=WHITE)


# ====================== SECTION 7: INTERVIEW Q1 ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "7", "⑦ 面試加分洞察 Q1：CSP CapEx 持續成長，誰是最大受惠者？", 18, 25)
add_rect(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(6.0), LIGHT, line=NAVY)
add_text(s, Inches(0.6), Inches(1.3), Inches(12), Inches(0.5),
         "▸  研究員級回答 — 分『絕對受惠』與『風險調整後受惠』兩層", size=14, bold=True, color=NAVY)

add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(0.5),
         "1️⃣  絕對金額最大受惠者", size=13, bold=True, color=GOLD)
add_text(s, Inches(1.0), Inches(2.45), Inches(11.7), Inches(1.0),
         "NVIDIA——拿走 AI 伺服器 BOM 約七成。FY26 資料中心營收 $1,937 億 (+68%)，2026/4 季年化逾 $3,000 億。",
         size=12, color=DARK_TEXT)

add_text(s, Inches(0.7), Inches(3.5), Inches(12), Inches(0.5),
         "2️⃣  風險調整後最佳的受惠者 (研究員真正關注)", size=13, bold=True, color=GOLD)
add_text(s, Inches(1.0), Inches(3.95), Inches(11.7), Inches(1.0),
         ["『不可替代的瓶頸』——才有持續定價權：",
          "• 台積電 CoWoS：連 NVIDIA 都要排隊",
          "• SK 海力士 HBM：2026 產能全數售罄"],
         size=12, color=DARK_TEXT)

add_text(s, Inches(0.7), Inches(5.4), Inches(12), Inches(0.5),
         "3️⃣  台股最佳對應", size=13, bold=True, color=GOLD)
add_text(s, Inches(1.0), Inches(5.85), Inches(11.7), Inches(1.2),
         ["• 台積電 2330 (CoWoS)、台達電 2308 (電源)、雙鴻/奇鋐 (液冷)——共同點：規格升級帶動 ASP，且競爭格局集中",
          "• 避開：純做整機、毛利率被 CSP 壓縮的同質代工，除非在液冷機櫃整合上有差異化"],
         size=12, color=DARK_TEXT)


# ====================== SECTION 7: INTERVIEW Q2 ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "7", "⑦ 面試加分洞察 Q2：未來三年最值得追蹤的環節？", 19, 25)
add_rect(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(6.0), LIGHT, line=NAVY)
add_text(s, Inches(0.6), Inches(1.3), Inches(12), Inches(0.5),
         "▸  答案：電力與先進封裝 — 邏輯是『瓶頸會外溢』", size=14, bold=True, color=NAVY)

phases = [
    ("2023–2025", "瓶頸 = GPU + HBM", NAVY),
    ("2026–2028", "外溢至 電力 (併網、變壓器、HVDC、液冷)\n＋ CoWoS / CPO (光電共封裝)", GOLD),
]
for i, (yr, desc, c) in enumerate(phases):
    y = Inches(2.1 + i * 1.5)
    add_rect(s, Inches(0.7), y, Inches(2.5), Inches(1.2), c)
    add_text(s, Inches(0.7), y + Inches(0.35), Inches(2.5), Inches(0.6),
             yr, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(3.4), y, Inches(9.4), Inches(1.2), WHITE, line=c)
    add_text(s, Inches(3.6), y + Inches(0.2), Inches(9.1), Inches(0.9),
             desc, size=13, color=DARK_TEXT)

add_text(s, Inches(0.7), Inches(5.4), Inches(12), Inches(0.5),
         "▸  我追蹤三個領先指標", size=13, bold=True, color=NAVY)
add_text(s, Inches(1.0), Inches(5.9), Inches(11.7), Inches(1.5),
         ["①  CoWoS 月產能擴張進度 (75K → 130K 片 / 月)",
          "②  資料中心電網併網排隊時間 ＋ 燃氣輪機 / SMR 交期",
          "③  1.6T 光模組與 EML 雷射缺口 — NVIDIA $40 億鎖定 Lumentum/Coherent 是鐵證"],
         size=12, color=DARK_TEXT)


# ====================== SECTION 7: INTERVIEW Q3 ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "7", "⑦ 面試加分洞察 Q3：若 CapEx 放緩，誰先受衝擊？", 20, 25)
add_rect(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(6.0), LIGHT, line=NAVY)
add_text(s, Inches(0.6), Inches(1.3), Inches(12), Inches(0.5),
         "▸  排序原則：離終端需求最遠 × 財務最脆弱 × 議價力最弱", size=14, bold=True, color=NAVY)

order = [
    ("①  最先受衝擊", "高槓桿 neocloud (CoreWeave 等)", "GPU 抵押融資對利用率與殘值極度敏感", RED),
    ("②  次受衝擊", "低毛利伺服器 ODM", "訂單能見度短、規格急變、CSP 兩面議價", GOLD),
    ("③  其次", "過剩世代 GPU 租賃", "舊款 H100 租金已在鬆動", GOLD),
    ("④  最後才會傷到", "瓶頸環節 (CoWoS / HBM / 電力)", "賣方市場、訂單能見度長", GREEN),
]
for i, (rank, name, why, c) in enumerate(order):
    y = Inches(2.05 + i * 0.95)
    add_rect(s, Inches(0.6), y, Inches(2.4), Inches(0.85), c)
    add_text(s, Inches(0.6), y + Inches(0.18), Inches(2.4), Inches(0.5),
             rank, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(3.2), y, Inches(9.6), Inches(0.85), WHITE, line=c)
    add_text(s, Inches(3.4), y + Inches(0.08), Inches(4.0), Inches(0.7),
             name, size=13, bold=True, color=NAVY)
    add_text(s, Inches(7.6), y + Inches(0.08), Inches(5.0), Inches(0.7),
             why, size=11, color=DARK_TEXT)

add_rect(s, Inches(0.6), Inches(6.0), Inches(12.2), Inches(1.0), NAVY)
add_text(s, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.4),
         "▸  操作策略：做多瓶頸、放空過剩 — 而非一刀切看空整條供應鏈", size=13, bold=True, color=GOLD)
add_text(s, Inches(0.8), Inches(6.55), Inches(11.8), Inches(0.5),
         "▸  CapEx 見頂的『語言訊號』：CSP 法說會口徑由『supply-constrained』轉為『optimizing utilization』",
         size=11, color=WHITE)


# ====================== APPENDIX: KEY DATA POINTS ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "A", "Appendix A：關鍵數據錨點（記憶用）", 21, 25)
data = [
    ["主題", "錨點數據", "來源"],
    ["四大 2025 CapEx", "~$388–410B (YoY +73%)", "CNBC / Tom's Hardware"],
    ["四大 2026E CapEx", "~$630–725B (+60~77%)", "公司財報指引"],
    ["NVIDIA FY26 DC 營收", "$193.7B (+68%); Q1 FY27 $75.2B (年化 ~$300B)", "NVIDIA 8-K"],
    ["Oracle RPO", "$98B (FY24) → $638B (FY26, +363%)", "Oracle 8-K"],
    ["B200 BOM 拆解", "HBM 45% / 邏輯 22% / CoWoS 17% (~$6,400 製造成本)", "Epoch AI"],
    ["Rubin VR200 機櫃", "~$7.8M；HBM4+LPDDR5X 達 $2M (25%)", "Morgan Stanley"],
    ["HBM 市場", "$18.2B (24) → $46.7B (25, +156%); SK / Micron 2026 售罄", "TrendForce"],
    ["CoWoS 產能", "3.5–4 萬 (24) → 7.5 萬 (25) → 12–13 萬 (26 目標) 片/月", "TrendForce"],
    ["AI 伺服器市場", "$187B (24) → $298B (25) → 佔總伺服器 74% (26)", "TrendForce"],
    ["DC 用電", "2025 +17% YoY；2030 翻倍至 945 TWh", "IEA"],
    ["設備交期", "變壓器 128 週、GSU 143 週、燃氣輪機 7 年", "Power Magazine / GE Vernova"],
    ["800G+ 出貨", "2,400 萬 (25) → 6,300 萬顆 (26, +2.6x)", "TrendForce / LightCounting"],
]
add_table(s, Inches(0.3), Inches(1.05), Inches(12.7), Inches(6.0), data,
          col_widths=[2.6, 7.5, 2.6], first_col_bold=True,
          body_size=11, header_size=12)


# ====================== APPENDIX: SOURCES ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "A", "Appendix B：主要資料來源", 22, 25)

add_rect(s, Inches(0.4), Inches(1.1), Inches(6.1), Inches(6.0), LIGHT, line=NAVY)
add_text(s, Inches(0.6), Inches(1.25), Inches(5.7), Inches(0.4),
         "▸  第一手 (Primary)", size=14, bold=True, color=NAVY)
add_text(s, Inches(0.6), Inches(1.7), Inches(5.7), Inches(5.3),
         ["• Amazon 10-K / Q4 法說會",
          "• Microsoft 8-K (FY2023–FY2026)",
          "• Alphabet 8-K / 10-K (FY2023–FY2026)",
          "• Meta 8-K (FY2024)",
          "• Oracle 8-K (FY24–FY26, RPO/CapEx)",
          "• NVIDIA Q3/Q4 FY26 8-K, Q1 FY27 8-K",
          "• Broadcom / Marvell / AMD / Lumentum 8-K",
          "• TSMC Q1 2026 法說會 (CapEx & CoWoS)",
          "• Vertiv / Eaton / GE Vernova 財報 / IR"],
         size=11, color=DARK_TEXT)

add_rect(s, Inches(6.7), Inches(1.1), Inches(6.2), Inches(6.0), LIGHT, line=NAVY)
add_text(s, Inches(6.9), Inches(1.25), Inches(5.7), Inches(0.4),
         "▸  第二手 (Industry Research)", size=14, bold=True, color=NAVY)
add_text(s, Inches(6.9), Inches(1.7), Inches(5.7), Inches(5.3),
         ["• TrendForce：HBM / NAND / CoWoS / AI 伺服器",
          "• Dell'Oro：DC 交換器 / 液冷 / DCPI",
          "• LightCounting：光收發模組市場",
          "• Counterpoint：HBM / DRAM 市占",
          "• SemiAnalysis：AI 伺服器成本拆解",
          "• Epoch AI：B200 / 1GW DC 成本模型",
          "• IEA / EPRI：資料中心用電預測",
          "• Goldman Sachs / Morgan Stanley：估值與 BOM",
          "• McKinsey：2030 $6.7T 資料中心累計需求",
          "• CNBC、Tom's Hardware、Sherwood News"],
         size=11, color=DARK_TEXT)


# ====================== DISCLAIMER ======================
s = prs.slides.add_slide(blank)
add_slide_header(s, "A", "Appendix C：信心分級與免責聲明", 23, 25)
data = [
    ["信心分級", "適用範圍", "建議讀法"],
    ["① 高信心 (High)", "SEC 財報絕對值、Oracle RPO、HBM/NAND 售罄狀態、各公司 2026 指引、NVIDIA DC 營收",
     "可直接引用"],
    ["② 中信心 (Medium)", "市占率百分比、BOM 拆解占比、CoWoS 產能數字 (跨來源估算)",
     "以 ±10–20% 區間理解"],
    ["③ 估算 (Estimate)", "CapEx/Revenue、CapEx/OCF 比率、2026 FCF 預估 (分析師模型)",
     "為情境推演用、非公司公告"],
]
add_table(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(3.0), data,
          col_widths=[2.2, 7.5, 2.8], first_col_bold=True,
          body_size=11, header_size=12)

add_rect(s, Inches(0.4), Inches(4.4), Inches(12.5), Inches(2.6), NAVY)
add_text(s, Inches(0.6), Inches(4.6), Inches(12.1), Inches(0.5),
         "DISCLAIMER  免責聲明", size=14, bold=True, color=GOLD)
add_text(s, Inches(0.6), Inches(5.1), Inches(12.1), Inches(1.9),
         ["• 本簡報為產業研究與投資分析用途，所有前瞻數字 (2026E) 為公司指引或分析師估計，非實際結果。",
          "• 本簡報不構成任何投資建議；個股提及僅為產業關聯說明，不代表買賣建議。",
          "• 部分原始機構頁面 (TrendForce / Dell'Oro / SemiAnalysis 等) 無法直接自動抓取，",
          "  相關數字係透過多方來源交叉佐證，正式對外引用前建議再以原始 PDF / 財報核對。",
          "• 報告完成日：2026 年 6 月。"],
         size=11, color=WHITE)


# ====================== THANK YOU ======================
s = prs.slides.add_slide(blank)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, NAVY)
add_rect(s, 0, Inches(3.2), SLIDE_W, Inches(0.05), GOLD)
add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(0.5),
         "INSTITUTIONAL EQUITY RESEARCH", size=14, bold=True, color=GOLD)
add_text(s, Inches(0.7), Inches(2.7), Inches(12), Inches(1.2),
         "Thank You.", size=48, bold=True, color=WHITE)
add_text(s, Inches(0.7), Inches(3.6), Inches(12), Inches(0.5),
         "Q & A", size=24, color=GOLD)
add_text(s, Inches(0.7), Inches(4.6), Inches(12), Inches(2.0),
         ["7 部分  ·  25 張投影片  ·  涵蓋 5 家 CSP + 全球與台灣供應鏈",
          "",
          "本研究報告 Markdown 全文已同步推送至 Git 分支：",
          "    claude/trusting-thompson-f9wa47",
          "",
          "建議搭配：個股估值試算  ·  Bull/Base/Bear EPS 推演  ·  目標價設定"],
         size=14, color=WHITE)

out = "/home/user/-/全球CSP_CapEx_深度研究簡報.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Total slides: {len(prs.slides)}")
