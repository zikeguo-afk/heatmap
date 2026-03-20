const imageInput = document.getElementById("imageInput");
const promptInput = document.getElementById("promptInput");
const viewSelect = document.getElementById("viewSelect");
const generateBtn = document.getElementById("generateBtn");
const demoBtn = document.getElementById("demoBtn");
const resetBtn = document.getElementById("resetBtn");
const fileName = document.getElementById("fileName");
const tokenList = document.getElementById("tokenList");
const threshold = document.getElementById("threshold");
const thresholdValue = document.getElementById("thresholdValue");
const opacity = document.getElementById("opacity");
const opacityValue = document.getElementById("opacityValue");
const stepSlider = document.getElementById("stepSlider");
const stepValue = document.getElementById("stepValue");
const overlayToggle = document.getElementById("overlayToggle");
const playBtn = document.getElementById("playBtn");
const summaryText = document.getElementById("summaryText");
const promptText = document.getElementById("promptText");
const selectedTokenLabel = document.getElementById("selectedTokenLabel");
const hint = document.getElementById("hint");
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");

const MAX_DIMENSION = 960;
const TOTAL_STEPS = 12;
const STOPWORDS = new Set([
  "a",
  "an",
  "the",
  "with",
  "and",
  "or",
  "of",
  "for",
  "to",
  "in",
  "on",
  "at",
  "by",
  "car's",
  "的",
  "和",
  "与",
  "并且",
  "以及",
  "一个",
  "一辆",
]);

const state = {
  baseSource: null,
  width: 900,
  height: 540,
  prompt: "",
  tokens: [],
  selectedTokenId: null,
  heatmapCache: new Map(),
  animationTimer: null,
};

const RULES = [
  {
    id: "wheels",
    aliases: ["wheel", "wheels", "tire", "tires", "rim", "rims", "轮胎", "轮毂", "车轮", "轮子"],
    matched: true,
    paint(map, layout) {
      paintWheelPair(map, layout, 1.0);
      paintBlobs(map, layout, [
        [layout.rearWheelX, layout.wheelY - 0.07, 0.07, 0.06, 0.42],
        [layout.frontWheelX, layout.wheelY - 0.07, 0.07, 0.06, 0.42],
      ]);
    },
  },
  {
    id: "headlights",
    aliases: ["headlight", "headlights", "lamp", "led", "frontlight", "前灯", "头灯", "大灯", "灯组", "日行灯"],
    matched: true,
    paint(map, layout) {
      const points = layout.frontView
        ? [
            [0.34, 0.42, 0.05, 0.035, 0.95],
            [0.66, 0.42, 0.05, 0.035, 0.95],
          ]
        : [
            [layout.frontX, layout.lightY, 0.075, 0.042, 1.0],
            [layout.frontX - layout.frontDirection * 0.05, layout.lightY + 0.01, 0.07, 0.035, 0.58],
          ];
      paintBlobs(map, layout, points);
    },
  },
  {
    id: "taillights",
    aliases: ["taillight", "taillights", "rearlight", "尾灯", "贯穿尾灯", "尾部灯带"],
    matched: true,
    paint(map, layout) {
      const points = layout.rearView
        ? [
            [0.34, 0.42, 0.05, 0.035, 0.9],
            [0.66, 0.42, 0.05, 0.035, 0.9],
          ]
        : [
            [layout.rearX, layout.lightY + 0.01, 0.072, 0.04, 0.94],
            [layout.rearX + layout.frontDirection * 0.045, layout.lightY + 0.02, 0.058, 0.036, 0.55],
          ];
      paintBlobs(map, layout, points);
    },
  },
  {
    id: "roof",
    aliases: ["roof", "top", "roofline", "panoramic", "panorama", "车顶", "顶棚", "天幕", "全景天幕", "全景"],
    matched: true,
    paint(map, layout) {
      paintLine(map, layout, layout.rearRoofX, layout.roofY, layout.frontRoofX, layout.roofY - 0.01, 0.05, 0.9, 18);
      paintBlobs(map, layout, [[layout.centerX, layout.roofY + 0.01, 0.13, 0.04, 0.46]]);
    },
  },
  {
    id: "windows",
    aliases: ["window", "windows", "glass", "windshield", "车窗", "玻璃", "挡风玻璃", "侧窗"],
    matched: true,
    paint(map, layout) {
      paintLine(map, layout, layout.rearCabinX, layout.cabinY, layout.frontCabinX, layout.cabinY - 0.005, 0.06, 0.78, 16);
      paintBlobs(map, layout, [
        [layout.centerX, layout.cabinY + 0.02, 0.12, 0.05, 0.55],
        [layout.frontCabinX, layout.cabinY, 0.08, 0.05, 0.42],
      ]);
    },
  },
  {
    id: "hood",
    aliases: ["hood", "bonnet", "front cover", "引擎盖", "前舱盖", "机盖"],
    matched: true,
    paint(map, layout) {
      paintLine(map, layout, layout.centerX + layout.frontDirection * 0.07, layout.hoodY, layout.frontX, layout.hoodY + 0.03, 0.06, 0.92, 14);
      paintBlobs(map, layout, [[layout.frontX - layout.frontDirection * 0.06, layout.hoodY + 0.02, 0.1, 0.06, 0.45]]);
    },
  },
  {
    id: "grille",
    aliases: ["grille", "front face", "nose", "clean", "minimal", "中网", "前脸", "格栅", "鼻翼", "简洁", "极简"],
    matched: true,
    paint(map, layout) {
      if (layout.frontView) {
        paintBlobs(map, layout, [
          [0.5, 0.53, 0.16, 0.11, 0.95],
          [0.5, 0.42, 0.12, 0.06, 0.38],
        ]);
        return;
      }
      paintBlobs(map, layout, [[layout.frontX, 0.55, 0.085, 0.09, 0.95]]);
      paintLine(map, layout, layout.frontX - layout.frontDirection * 0.045, 0.5, layout.frontX - layout.frontDirection * 0.03, 0.61, 0.04, 0.42, 8);
    },
  },
  {
    id: "bumper",
    aliases: ["bumper", "splitter", "diffuser", "保险杠", "前杠", "后杠", "下包围"],
    matched: true,
    paint(map, layout) {
      if (layout.frontView || layout.rearView) {
        paintLine(map, layout, 0.28, 0.68, 0.72, 0.68, 0.05, 0.92, 14);
        return;
      }
      paintBlobs(map, layout, [
        [layout.frontX, 0.67, 0.09, 0.06, 0.82],
        [layout.rearX, 0.68, 0.08, 0.06, 0.52],
      ]);
    },
  },
  {
    id: "doors",
    aliases: ["door", "doors", "handle", "handles", "车门", "门把手", "侧门"],
    matched: true,
    paint(map, layout) {
      paintLine(map, layout, layout.rearDoorX, 0.57, layout.frontDoorX, 0.57, 0.06, 0.82, 18);
      paintBlobs(map, layout, [
        [layout.centerX - 0.08, 0.57, 0.07, 0.08, 0.4],
        [layout.centerX + 0.08, 0.57, 0.07, 0.08, 0.4],
      ]);
    },
  },
  {
    id: "mirrors",
    aliases: ["mirror", "mirrors", "后视镜", "外后视镜"],
    matched: true,
    paint(map, layout) {
      if (layout.frontView || layout.rearView) {
        paintBlobs(map, layout, [
          [0.24, 0.39, 0.035, 0.05, 0.8],
          [0.76, 0.39, 0.035, 0.05, 0.8],
        ]);
        return;
      }
      paintBlobs(map, layout, [[layout.frontCabinX - layout.frontDirection * 0.02, 0.42, 0.04, 0.05, 0.88]]);
    },
  },
  {
    id: "spoiler",
    aliases: ["spoiler", "wing", "尾翼", "鸭尾"],
    matched: true,
    paint(map, layout) {
      paintBlobs(map, layout, [[layout.rearX, layout.roofY + 0.02, 0.06, 0.03, 0.92]]);
      paintLine(map, layout, layout.rearX, layout.roofY + 0.02, layout.rearX + layout.frontDirection * 0.06, layout.roofY + 0.01, 0.03, 0.4, 6);
    },
  },
  {
    id: "fender",
    aliases: ["fender", "fenders", "arch", "arches", "翼子板", "轮眉"],
    matched: true,
    paint(map, layout) {
      paintBlobs(map, layout, [
        [layout.rearWheelX, layout.wheelY - 0.06, 0.09, 0.06, 0.9],
        [layout.frontWheelX, layout.wheelY - 0.06, 0.09, 0.06, 0.9],
      ]);
    },
  },
  {
    id: "body",
    aliases: ["car", "vehicle", "body", "shape", "silhouette", "车", "车身", "整车", "轮廓", "外形"],
    matched: true,
    paint(map, layout) {
      paintBody(map, layout, 1.0);
    },
  },
  {
    id: "futuristic",
    aliases: ["futuristic", "future", "concept", "concept car", "未来感", "未来", "概念车", "科幻"],
    matched: true,
    paint(map, layout) {
      paintBody(map, layout, 0.5);
      RULES_BY_ID.lightbar.paint(map, layout);
      RULES_BY_ID.roof.paint(map, layout);
      paintBlobs(map, layout, [
        [layout.frontWheelX, layout.wheelY, 0.09, 0.09, 0.35],
        [layout.rearWheelX, layout.wheelY, 0.09, 0.09, 0.35],
      ]);
    },
  },
  {
    id: "electric",
    aliases: ["electric", "ev", "新能源", "电动", "纯电"],
    matched: true,
    paint(map, layout) {
      RULES_BY_ID.lightbar.paint(map, layout);
      RULES_BY_ID.grille.paint(map, layout);
      paintBlobs(map, layout, [[layout.centerX, 0.59, 0.16, 0.08, 0.42]]);
    },
  },
  {
    id: "sporty",
    aliases: ["sport", "sporty", "aggressive", "sharp", "dynamic", "运动", "激进", "锐利", "锋利"],
    matched: true,
    paint(map, layout) {
      RULES_BY_ID.headlights.paint(map, layout);
      RULES_BY_ID.hood.paint(map, layout);
      paintLine(map, layout, layout.centerX - 0.06, 0.62, layout.frontX, 0.58, 0.05, 0.6, 12);
    },
  },
  {
    id: "luxury",
    aliases: ["luxury", "premium", "elegant", "豪华", "高级", "精致"],
    matched: true,
    paint(map, layout) {
      RULES_BY_ID.grille.paint(map, layout);
      RULES_BY_ID.windows.paint(map, layout);
      paintBlobs(map, layout, [[layout.centerX, 0.55, 0.18, 0.12, 0.34]]);
    },
  },
  {
    id: "suv",
    aliases: ["suv", "offroad", "crossover", "large", "wide", "big", "越野", "suv车型", "跨界", "大尺寸", "宽体"],
    matched: true,
    paint(map, layout) {
      paintBody(map, layout, 0.72, 0.06);
      paintWheelPair(map, layout, 0.8, 0.025);
      paintBlobs(map, layout, [[layout.centerX, 0.44, 0.2, 0.09, 0.42]]);
    },
  },
  {
    id: "sedan",
    aliases: ["sedan", "轿车", "三厢"],
    matched: true,
    paint(map, layout) {
      paintBody(map, layout, 0.7, -0.015);
      paintLine(map, layout, layout.rearRoofX + 0.02, layout.roofY, layout.centerX + 0.1, 0.38, 0.05, 0.62, 10);
    },
  },
  {
    id: "coupe",
    aliases: ["coupe", "fastback", "溜背", "跑车", "双门"],
    matched: true,
    paint(map, layout) {
      paintLine(map, layout, layout.rearRoofX + 0.06, layout.roofY - 0.01, layout.frontRoofX, layout.roofY + 0.005, 0.045, 0.86, 16);
      paintBody(map, layout, 0.65, -0.03);
    },
  },
  {
    id: "lightbar",
    aliases: ["lightbar", "strip", "灯带", "贯穿灯带", "贯穿式"],
    matched: true,
    paint(map, layout) {
      if (layout.frontView || layout.rearView) {
        paintLine(map, layout, 0.27, 0.44, 0.73, 0.44, 0.03, 0.95, 16);
        return;
      }
      paintLine(map, layout, layout.frontX - layout.frontDirection * 0.08, layout.lightY, layout.frontX + layout.frontDirection * 0.03, layout.lightY, 0.03, 0.95, 10);
      paintLine(map, layout, layout.rearX + layout.frontDirection * 0.01, layout.lightY + 0.01, layout.rearX + layout.frontDirection * 0.08, layout.lightY + 0.01, 0.03, 0.58, 8);
    },
  },
  {
    id: "fallback",
    aliases: [],
    matched: false,
    paint(map, layout) {
      paintBody(map, layout, 0.45);
      paintBlobs(map, layout, [[layout.centerX, 0.48, 0.16, 0.1, 0.35]]);
    },
  },
];

const RULES_BY_ID = Object.fromEntries(RULES.map((rule) => [rule.id, rule]));

function fitDimensions(width, height) {
  const scale = Math.min(1, MAX_DIMENSION / Math.max(width, height));
  return {
    width: Math.max(320, Math.round(width * scale)),
    height: Math.max(220, Math.round(height * scale)),
  };
}

function createWorkingCanvas(width, height) {
  const element = document.createElement("canvas");
  element.width = width;
  element.height = height;
  return element;
}

function createDemoCanvas() {
  const demo = createWorkingCanvas(1400, 840);
  const demoCtx = demo.getContext("2d");

  const sky = demoCtx.createLinearGradient(0, 0, 0, demo.height);
  sky.addColorStop(0, "#19314f");
  sky.addColorStop(0.52, "#0e1a2b");
  sky.addColorStop(1, "#08101a");
  demoCtx.fillStyle = sky;
  demoCtx.fillRect(0, 0, demo.width, demo.height);

  demoCtx.fillStyle = "rgba(84, 212, 194, 0.08)";
  demoCtx.beginPath();
  demoCtx.arc(250, 180, 180, 0, Math.PI * 2);
  demoCtx.fill();

  demoCtx.fillStyle = "rgba(255, 184, 79, 0.08)";
  demoCtx.beginPath();
  demoCtx.arc(1160, 160, 140, 0, Math.PI * 2);
  demoCtx.fill();

  demoCtx.fillStyle = "#10161f";
  demoCtx.fillRect(0, 610, demo.width, 230);

  demoCtx.strokeStyle = "rgba(255,255,255,0.08)";
  demoCtx.lineWidth = 4;
  demoCtx.beginPath();
  demoCtx.moveTo(120, 626);
  demoCtx.lineTo(1280, 626);
  demoCtx.stroke();

  demoCtx.fillStyle = "#d3d8de";
  demoCtx.beginPath();
  demoCtx.moveTo(250, 520);
  demoCtx.bezierCurveTo(360, 390, 560, 328, 780, 336);
  demoCtx.bezierCurveTo(920, 340, 1050, 392, 1170, 512);
  demoCtx.lineTo(1144, 588);
  demoCtx.lineTo(232, 588);
  demoCtx.closePath();
  demoCtx.fill();

  demoCtx.fillStyle = "#8aa4ba";
  demoCtx.beginPath();
  demoCtx.moveTo(472, 380);
  demoCtx.lineTo(758, 380);
  demoCtx.lineTo(910, 508);
  demoCtx.lineTo(364, 508);
  demoCtx.closePath();
  demoCtx.fill();

  demoCtx.fillStyle = "#9fcfd4";
  demoCtx.fillRect(1022, 478, 90, 20);
  demoCtx.fillRect(258, 490, 72, 18);

  demoCtx.strokeStyle = "#f8f8f6";
  demoCtx.lineWidth = 8;
  demoCtx.beginPath();
  demoCtx.moveTo(330, 588);
  demoCtx.lineTo(1080, 588);
  demoCtx.stroke();

  paintDemoWheel(demoCtx, 430, 590, 95);
  paintDemoWheel(demoCtx, 980, 590, 95);

  demoCtx.fillStyle = "#10161f";
  demoCtx.fillRect(600, 530, 152, 8);
  demoCtx.fillRect(518, 462, 10, 58);

  return demo;
}

function paintDemoWheel(demoCtx, x, y, radius) {
  demoCtx.fillStyle = "#0d1014";
  demoCtx.beginPath();
  demoCtx.arc(x, y, radius, 0, Math.PI * 2);
  demoCtx.fill();

  demoCtx.fillStyle = "#8995a0";
  demoCtx.beginPath();
  demoCtx.arc(x, y, radius * 0.56, 0, Math.PI * 2);
  demoCtx.fill();

  demoCtx.fillStyle = "#1d2730";
  demoCtx.beginPath();
  demoCtx.arc(x, y, radius * 0.22, 0, Math.PI * 2);
  demoCtx.fill();
}

function scaleSourceToCanvas(source, width, height) {
  const sized = createWorkingCanvas(width, height);
  const sizedCtx = sized.getContext("2d");
  sizedCtx.drawImage(source, 0, 0, width, height);
  return sized;
}

function setBaseSource(source, label) {
  const dimensions = fitDimensions(source.width, source.height);
  state.baseSource = scaleSourceToCanvas(source, dimensions.width, dimensions.height);
  state.width = dimensions.width;
  state.height = dimensions.height;
  canvas.width = state.width;
  canvas.height = state.height;
  fileName.textContent = label;
  render();
}

function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Image load failed"));
    };
    img.src = url;
  });
}

function normalizeToken(text) {
  const normalized = text.trim().toLowerCase();
  if (/^[a-z-]+$/.test(normalized)) {
    if (normalized.endsWith("ies") && normalized.length > 4) {
      return `${normalized.slice(0, -3)}y`;
    }
    if (/(ches|shes|xes|zes|ses)$/.test(normalized) && normalized.length > 4) {
      return normalized.slice(0, -2);
    }
    if (normalized.endsWith("s") && !normalized.endsWith("ss") && normalized.length > 4) {
      return normalized.slice(0, -1);
    }
  }
  return normalized;
}

function extractPromptTokens(prompt) {
  const original = prompt.trim();
  if (!original) {
    return [];
  }

  const normalized = original.toLowerCase();
  const hits = [];

  RULES.forEach((rule) => {
    rule.aliases.forEach((alias) => {
      if (!alias) {
        return;
      }
      const query = alias.toLowerCase();
      let start = normalized.indexOf(query);
      while (start !== -1) {
        hits.push({
          raw: alias,
          normalized: normalizeToken(query),
          index: start,
          categoryIds: [rule.id],
          matched: rule.matched,
        });
        start = normalized.indexOf(query, start + query.length);
      }
    });
  });

  const freeTokens = [];
  const regex = /[A-Za-z0-9-]+|[\u4e00-\u9fff]{1,10}/g;
  for (const match of original.matchAll(regex)) {
    const raw = match[0];
    freeTokens.push({
      raw,
      normalized: normalizeToken(raw),
      index: match.index || 0,
      categoryIds: [],
      matched: false,
    });
  }

  const merged = [...hits, ...freeTokens].sort((a, b) => a.index - b.index || a.raw.length - b.raw.length);
  const deduped = [];
  const seen = new Map();

  for (const item of merged) {
    if (!item.normalized) {
      continue;
    }
    if (STOPWORDS.has(item.normalized)) {
      continue;
    }

    const existingIndex = seen.get(item.normalized);
    if (existingIndex !== undefined) {
      const existing = deduped[existingIndex];
      existing.matched = existing.matched || item.matched;
      existing.categoryIds = Array.from(new Set([...existing.categoryIds, ...item.categoryIds]));
      continue;
    }

    const categoryIds = item.categoryIds.length ? item.categoryIds : matchRuleIds(item.normalized);
    deduped.push({
      id: `${item.normalized}-${deduped.length}`,
      text: item.raw,
      normalized: item.normalized,
      matched: categoryIds.length > 0,
      categoryIds: categoryIds.length ? categoryIds : ["fallback"],
    });
    seen.set(item.normalized, deduped.length - 1);
  }

  return deduped.slice(0, 18);
}

function matchRuleIds(token) {
  const minimumLength = /[\u4e00-\u9fff]/.test(token) ? 2 : 3;
  return RULES
    .filter((rule) =>
      rule.aliases.some((alias) => {
        const normalizedAlias = alias.toLowerCase();
        if (normalizedAlias === token) {
          return true;
        }
        if (token.length < minimumLength) {
          return false;
        }
        return normalizedAlias.includes(token) || token.includes(normalizedAlias);
      }),
    )
    .map((rule) => rule.id)
    .filter((id) => id !== "fallback");
}

function createLayout(view) {
  const base = {
    centerX: 0.5,
    roofY: 0.27,
    cabinY: 0.36,
    hoodY: 0.41,
    wheelY: 0.77,
    lightY: 0.5,
    frontView: false,
    rearView: false,
  };

  if (view === "front") {
    return {
      ...base,
      frontView: true,
      rearView: false,
      frontDirection: 1,
      frontX: 0.5,
      rearX: 0.5,
      rearWheelX: 0.34,
      frontWheelX: 0.66,
      rearRoofX: 0.34,
      frontRoofX: 0.66,
      rearCabinX: 0.37,
      frontCabinX: 0.63,
      rearDoorX: 0.36,
      frontDoorX: 0.64,
    };
  }

  if (view === "rear") {
    return {
      ...createLayout("front"),
      rearView: true,
      frontView: false,
    };
  }

  if (view === "side-left") {
    return {
      ...base,
      frontDirection: -1,
      frontX: 0.21,
      rearX: 0.8,
      rearWheelX: 0.72,
      frontWheelX: 0.3,
      rearRoofX: 0.74,
      frontRoofX: 0.38,
      rearCabinX: 0.67,
      frontCabinX: 0.42,
      rearDoorX: 0.64,
      frontDoorX: 0.38,
    };
  }

  if (view === "three-quarter") {
    return {
      ...base,
      frontDirection: 1,
      frontX: 0.72,
      rearX: 0.29,
      rearWheelX: 0.34,
      frontWheelX: 0.69,
      rearRoofX: 0.39,
      frontRoofX: 0.64,
      rearCabinX: 0.45,
      frontCabinX: 0.61,
      rearDoorX: 0.41,
      frontDoorX: 0.61,
    };
  }

  return {
    ...base,
    frontDirection: 1,
    frontX: 0.79,
    rearX: 0.2,
    rearWheelX: 0.29,
    frontWheelX: 0.71,
    rearRoofX: 0.37,
    frontRoofX: 0.63,
    rearCabinX: 0.42,
    frontCabinX: 0.59,
    rearDoorX: 0.38,
    frontDoorX: 0.62,
  };
}

function paintBody(map, layout, strength = 1, yOffset = 0) {
  paintLine(map, layout, layout.rearX + 0.02, 0.57 + yOffset, layout.frontX - 0.02, 0.57 + yOffset, 0.11, 0.92 * strength, 18);
  paintLine(map, layout, layout.rearRoofX, layout.roofY + yOffset, layout.frontRoofX, layout.roofY + yOffset, 0.07, 0.58 * strength, 14);
  paintBlobs(map, layout, [
    [layout.centerX, 0.56 + yOffset, 0.22, 0.13, 0.64 * strength],
    [layout.centerX, 0.46 + yOffset, 0.18, 0.08, 0.42 * strength],
  ]);
}

function paintWheelPair(map, layout, strength = 1, sizeBoost = 0) {
  paintBlobs(map, layout, [
    [layout.rearWheelX, layout.wheelY, 0.08 + sizeBoost, 0.08 + sizeBoost, 1.0 * strength],
    [layout.frontWheelX, layout.wheelY, 0.08 + sizeBoost, 0.08 + sizeBoost, 1.0 * strength],
  ]);
}

function paintBlobs(map, layout, blobs) {
  blobs.forEach(([x, y, sx, sy, weight]) => {
    paintBlob(map, layout, x, y, sx, sy, weight);
  });
}

function paintBlob(map, layout, cx, cy, sigmaX, sigmaY, weight) {
  const width = map.width;
  const height = map.height;
  const xMin = Math.max(0, Math.floor((cx - sigmaX * 3) * width));
  const xMax = Math.min(width - 1, Math.ceil((cx + sigmaX * 3) * width));
  const yMin = Math.max(0, Math.floor((cy - sigmaY * 3) * height));
  const yMax = Math.min(height - 1, Math.ceil((cy + sigmaY * 3) * height));

  for (let y = yMin; y <= yMax; y += 1) {
    const dy = (y / height - cy) / sigmaY;
    for (let x = xMin; x <= xMax; x += 1) {
      const dx = (x / width - cx) / sigmaX;
      const value = weight * Math.exp(-0.5 * (dx * dx + dy * dy));
      const index = y * width + x;
      map.data[index] = Math.max(map.data[index], value);
    }
  }
}

function paintLine(map, layout, x1, y1, x2, y2, sigma, weight, steps) {
  for (let step = 0; step < steps; step += 1) {
    const t = steps === 1 ? 0 : step / (steps - 1);
    const x = x1 + (x2 - x1) * t;
    const y = y1 + (y2 - y1) * t;
    paintBlob(map, layout, x, y, sigma, sigma * 0.7, weight);
  }
}

function buildHeatmapForToken(token, view) {
  const key = `${token.id}:${view}:${state.width}x${state.height}`;
  const cached = state.heatmapCache.get(key);
  if (cached) {
    return cached;
  }

  const layout = createLayout(view);
  const map = {
    width: state.width,
    height: state.height,
    data: new Float32Array(state.width * state.height),
  };

  token.categoryIds.forEach((id) => {
    const rule = RULES_BY_ID[id] || RULES_BY_ID.fallback;
    rule.paint(map, layout, state.prompt);
  });

  normalizeMap(map.data);
  state.heatmapCache.set(key, map);
  return map;
}

function normalizeMap(data) {
  let max = 0;
  for (let i = 0; i < data.length; i += 1) {
    if (data[i] > max) {
      max = data[i];
    }
  }
  if (max < 1e-6) {
    return;
  }
  for (let i = 0; i < data.length; i += 1) {
    data[i] /= max;
  }
}

function updateSliders() {
  thresholdValue.textContent = Number(threshold.value).toFixed(2);
  opacityValue.textContent = Number(opacity.value).toFixed(2);
  stepValue.textContent = `${stepSlider.value} / ${TOTAL_STEPS}`;
}

function getSelectedToken() {
  return state.tokens.find((token) => token.id === state.selectedTokenId) || null;
}

function renderTokens() {
  tokenList.innerHTML = "";

  if (!state.tokens.length) {
    const empty = document.createElement("div");
    empty.className = "meta-mini";
    empty.textContent = "还没有 token。";
    tokenList.appendChild(empty);
    return;
  }

  state.tokens.forEach((token) => {
    const button = document.createElement("button");
    button.className = "token-btn";
    if (!token.matched) {
      button.classList.add("weak");
    }
    if (token.id === state.selectedTokenId) {
      button.classList.add("active");
    }
    button.textContent = token.text;
    button.addEventListener("click", () => {
      state.selectedTokenId = token.id;
      selectedTokenLabel.textContent = token.text;
      hint.style.display = "none";
      renderTokens();
      render();
    });
    tokenList.appendChild(button);
  });
}

function buildSummary(tokens) {
  if (!tokens.length) {
    return "没有可解析的 token，请输入更具体的汽车造型提示词。";
  }
  const strong = tokens.filter((token) => token.matched).length;
  const weak = tokens.length - strong;
  return `解析出 ${tokens.length} 个 token，其中 ${strong} 个命中汽车语义词典，${weak} 个回退到车身弱热区。`;
}

function animationValue(baseValue, progress) {
  const focused = Math.pow(baseValue, 3.1 - progress * 2.1);
  const gate = 0.74 - progress * 0.58;
  const core = focused <= gate ? 0 : (focused - gate) / Math.max(1e-6, 1 - gate);
  const halo = Math.max(0, baseValue - gate * 0.62) * 0.38 * progress;
  return Math.min(1, Math.max(core, halo));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorMap(value) {
  const stops = [
    { t: 0.0, c: [12, 17, 26] },
    { t: 0.22, c: [35, 76, 151] },
    { t: 0.5, c: [84, 212, 194] },
    { t: 0.78, c: [255, 184, 79] },
    { t: 1.0, c: [246, 105, 107] },
  ];

  for (let i = 0; i < stops.length - 1; i += 1) {
    const current = stops[i];
    const next = stops[i + 1];
    if (value >= current.t && value <= next.t) {
      const local = (value - current.t) / (next.t - current.t);
      return [
        Math.round(lerp(current.c[0], next.c[0], local)),
        Math.round(lerp(current.c[1], next.c[1], local)),
        Math.round(lerp(current.c[2], next.c[2], local)),
      ];
    }
  }

  return stops[stops.length - 1].c;
}

function renderBase() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.baseSource) {
    ctx.drawImage(state.baseSource, 0, 0, canvas.width, canvas.height);
    return;
  }

  ctx.fillStyle = "#09111a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function renderOverlay(token) {
  if (!overlayToggle.checked || !token || !state.baseSource) {
    return;
  }

  const heatmap = buildHeatmapForToken(token, viewSelect.value);
  const overlayCanvas = createWorkingCanvas(state.width, state.height);
  const overlayCtx = overlayCanvas.getContext("2d");
  const imageData = overlayCtx.createImageData(state.width, state.height);
  const thresholdValueNumber = Number(threshold.value);
  const opacityValueNumber = Number(opacity.value);
  const progress = (Number(stepSlider.value) - 1) / (TOTAL_STEPS - 1);

  for (let i = 0; i < heatmap.data.length; i += 1) {
    const animated = animationValue(heatmap.data[i], progress);
    if (animated < thresholdValueNumber) {
      continue;
    }
    const [r, g, b] = colorMap(animated);
    const offset = i * 4;
    imageData.data[offset] = r;
    imageData.data[offset + 1] = g;
    imageData.data[offset + 2] = b;
    imageData.data[offset + 3] = Math.round(255 * opacityValueNumber * animated);
  }

  overlayCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(overlayCanvas, 0, 0);
}

function render() {
  renderBase();
  const token = getSelectedToken();
  renderOverlay(token);
}

function stopAnimation() {
  if (state.animationTimer) {
    window.clearInterval(state.animationTimer);
    state.animationTimer = null;
  }
  playBtn.textContent = "Play";
}

function toggleAnimation() {
  if (state.animationTimer) {
    stopAnimation();
    return;
  }
  if (!getSelectedToken()) {
    return;
  }

  playBtn.textContent = "Pause";
  state.animationTimer = window.setInterval(() => {
    const nextStep = Number(stepSlider.value) >= TOTAL_STEPS ? 1 : Number(stepSlider.value) + 1;
    stepSlider.value = String(nextStep);
    updateSliders();
    render();
  }, 160);
}

function applyPrompt() {
  const prompt = promptInput.value.trim();
  state.prompt = prompt;
  promptText.textContent = prompt || "未输入";
  state.heatmapCache.clear();

  const tokens = extractPromptTokens(prompt);
  state.tokens = tokens;
  state.selectedTokenId = tokens[0]?.id || null;
  selectedTokenLabel.textContent = tokens[0]?.text || "None";
  summaryText.textContent = buildSummary(tokens);
  renderTokens();

  if (tokens.length) {
    hint.style.display = "none";
  }

  render();
}

function resetApp() {
  stopAnimation();
  canvas.width = state.width;
  canvas.height = state.height;
  promptInput.value = "";
  fileName.textContent = "支持 JPG / PNG / WEBP，本地处理，不上传服务器。";
  promptText.textContent = "未输入";
  selectedTokenLabel.textContent = "None";
  summaryText.textContent = "等待图片和提示词。";
  tokenList.innerHTML = "";
  imageInput.value = "";
  state.baseSource = null;
  state.prompt = "";
  state.tokens = [];
  state.selectedTokenId = null;
  state.heatmapCache.clear();
  hint.style.display = "block";
  renderBase();
}

async function handleImageChange(file) {
  if (!file) {
    return;
  }
  const image = await readImageFile(file);
  setBaseSource(image, `已加载: ${file.name}`);
  hint.style.display = "block";
}

function loadDemo() {
  const demo = createDemoCanvas();
  setBaseSource(demo, "已加载: demo-car");
  promptInput.value = "futuristic electric SUV with sharp headlights, panoramic roof, large wheels and clean grille";
  applyPrompt();
  hint.style.display = "none";
}

imageInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }
  try {
    await handleImageChange(file);
  } catch (error) {
    summaryText.textContent = "图片加载失败，请更换文件。";
  }
});

generateBtn.addEventListener("click", () => {
  if (!state.baseSource) {
    summaryText.textContent = "请先上传汽车图片，或使用 demo car。";
    hint.style.display = "block";
    return;
  }
  stopAnimation();
  applyPrompt();
});

demoBtn.addEventListener("click", () => {
  stopAnimation();
  loadDemo();
});

resetBtn.addEventListener("click", () => {
  resetApp();
});

threshold.addEventListener("input", () => {
  updateSliders();
  render();
});

opacity.addEventListener("input", () => {
  updateSliders();
  render();
});

stepSlider.addEventListener("input", () => {
  updateSliders();
  render();
});

overlayToggle.addEventListener("change", render);
playBtn.addEventListener("click", toggleAnimation);
viewSelect.addEventListener("change", () => {
  state.heatmapCache.clear();
  render();
});

updateSliders();
resetApp();
