# 漢字部件拆解｜源流明體

以「源流明體」繪製的漢字部件分色教學頁：
<https://tyhcch1234.github.io/Chinese_Character/index.html>

## 內容

| 部分 | 說明 |
| --- | --- |
| 字形結構（9 種） | 獨體、左右、左中右、上下、上中下、品字形、全包圍、三面包圍、兩面包圍，各附一個示範字 |
| 詞語拆解（40 字） | 敏、終、於、操、場、籃、球、架、找、錶、喜、眉、梢、拿、幫、助、帶、焦、急、慰、慚、愧、面、耳、赤、温、暖、興、采、烈、跌、倒、默、作、聲、貪、玩、垂、喪、氣 |

顏色規則：

- <span>紅</span>：部首所在部件
- 藍 / 綠 / 橙：其餘部件

其他功能：按結構篩選、點按字形放大（← → 切換、Esc 關閉）、A4 列印排版、手機自適應。

## 檔案結構

```
index.html                  主頁（單檔，含 CSS / JS）
assets/chars/*.png          40 個字的部件分色圖（800×800）
assets/examples/*.png       9 個結構示範字
assets/GenRyuMin-subset.woff2  源流明體子集（約 80 KB）
chars.json / examples.json  頁面資料（結構、部首、部件）
_src/                       產生圖檔與頁面的腳本
```

## 部署（GitHub Pages）

1. 在 GitHub 建立 public repo：`Chinese_Character`
2. 把本資料夾內所有檔案上傳到 repo 根目錄（`index.html` 必須在根目錄）
3. repo → Settings → Pages → Source 選 `Deploy from a branch`，Branch 選 `main` / `(root)` → Save
4. 約 1 分鐘後開 <https://tyhcch1234.github.io/Chinese_Character/index.html>

用指令上傳：

```bash
git init -b main
git add .
git commit -m "漢字部件拆解：40 字部件分色 + 9 種結構示例"
git remote add origin https://github.com/tyhcch1234/Chinese_Character.git
git push -u origin main
```

## 字型

- 源流明體 GenRyuMin 2.000（ButTaiwan），授權 SIL Open Font License 1.1
- 原始專案：<https://github.com/ButTaiwan/genryu-font>
- 頁面只用 275 個字符的子集（`assets/GenRyuMin-subset.woff2`），其餘字元會退回系統明體。

## 重新產生

分色圖不是手繪，而是由字型輪廓計算出來：

1. 用 Pillow 以源流明體把字放大 1800px 點陣化
2. 以 8 連通標記連筆成「筆畫團」，再按字形結構的區域把每個筆畫團歸入所屬部件
3. 跨越部件邊界的筆畫團，逐像素按區域分色（邊界落在筆畫最窄處）
4. 縮至 800px、量化成 32 色 PNG

```bash
python3 -m pip install pillow numpy scipy fonttools brotli
# 先下載 GenRyuMin2TW-otf.zip 解壓到 _font/
python3 _src/build_parts.py     # 產生 assets/**/*.png、chars.json、examples.json
python3 _src/build_site.py      # 產生 index.html
```

字型子集：

```bash
python3 -m fontTools.subset _font/GenRyuMin2TW-M.otf \
  --text-file=_font/subset_text.txt --flavor=woff2 \
  --output-file=assets/GenRyuMin-subset.woff2 --layout-features='*' --no-hinting
```

## 備註

- 部件分色依字形結構自動切割，用於教學示意；少數字（如「幫」「興」「帶」）部件筆畫相連，分界線會落在筆畫最窄處，與傳統字書的部件歸屬可能略有出入。
- 圖檔可自由用於課堂教學。
