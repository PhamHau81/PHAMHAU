import { GoogleGenAI } from '@google/genai';
import { Question, ExamQuestion, GradeNumber } from '../types';

export interface AIModelConfig {
  id: string;
  name: string;
  badge: string;
  description: string;
}

/**
 * Danh sách Model Google Gemini chính thức được hỗ trợ 100% trên Google AI Studio.
 * Loại bỏ hoàn toàn các model giả định (như 2.5 flash / 3.0 preview) để tránh lỗi 404 NOT FOUND.
 */
export const AI_MODELS: AIModelConfig[] = [
  {
    id: 'gemini-2.0-flash',
    name: 'Gemini 2.0 Flash (Khuyên Dùng)',
    badge: 'Mặc định • Tốc độ cao & Ổn định nhất',
    description: 'Model thế hệ 2.0 mới nhất, tốc độ tạo đề cực nhanh, hỗ trợ JSON Schema chuẩn xác tuyệt đối.'
  },
  {
    id: 'gemini-1.5-flash',
    name: 'Gemini 1.5 Flash',
    badge: 'Tương thích cao • Quota dồi dào',
    description: 'Model tin cậy kinh điển, hạn ngạch miễn phí dồi dào (15 RPM), giải toán THPT chuẩn mực.'
  },
  {
    id: 'gemini-2.0-flash-lite',
    name: 'Gemini 2.0 Flash-Lite',
    badge: 'Siêu tốc độ • Phản hồi tức thì',
    description: 'Bản rút gọn siêu nhẹ, tối ưu hóa thời gian phản hồi cho các câu hỏi nhanh.'
  },
  {
    id: 'gemini-1.5-pro',
    name: 'Gemini 1.5 Pro',
    badge: 'Suy luận chuyên sâu',
    description: 'Mạnh mẽ trong việc phân tích các bài toán hình học không gian và xác suất thống kê phức tạp.'
  }
];

export const FALLBACK_CHAIN = [
  'gemini-2.0-flash',
  'gemini-1.5-flash',
  'gemini-2.0-flash-lite',
  'gemini-1.5-pro'
];

export const getStoredApiKey = (): string => {
  try {
    const local = localStorage.getItem('gemini_api_key');
    if (local && local.trim()) {
      return local.trim();
    }
  } catch {
    // ignore
  }

  // Tự động nhận diện API Key từ biến môi trường Vite nếu đã cấu hình (.env hoặc Vercel)
  try {
    const envKey =
      (import.meta as any).env?.VITE_GEMINI_API_KEY ||
      (import.meta as any).env?.GEMINI_API_KEY;
    if (envKey && typeof envKey === 'string' && envKey.trim()) {
      return envKey.trim();
    }
  } catch {
    // ignore
  }

  return '';
};

export const setStoredApiKey = (key: string): void => {
  try {
    localStorage.setItem('gemini_api_key', key.trim());
  } catch {
    // ignore
  }
};

export const getStoredModel = (): string => {
  try {
    const saved = localStorage.getItem('gemini_selected_model');
    // Nếu model đã lưu nằm trong danh sách hỗ trợ hoặc là model hợp lệ
    if (saved && AI_MODELS.some((m) => m.id === saved)) {
      return saved;
    }
  } catch {
    // ignore
  }
  return 'gemini-2.0-flash';
};

export const setStoredModel = (modelId: string): void => {
  try {
    localStorage.setItem('gemini_selected_model', modelId);
  } catch {
    // ignore
  }
};

interface CallAIOptions {
  prompt: string;
  systemInstruction?: string;
  responseMimeType?: string;
  responseSchema?: any;
  onModelAttempt?: (modelName: string) => void;
}

/**
 * Trích xuất text an toàn từ response của GoogleGenAI
 */
function extractResponseText(response: any): string {
  if (!response) return '';

  if (response.candidates && response.candidates[0]?.content?.parts) {
    const parts = response.candidates[0].content.parts;
    const textParts = parts.filter((p: any) => !p.thought && typeof p.text === 'string');
    if (textParts.length > 0) {
      return textParts.map((p: any) => p.text).join('').trim();
    }
    const allText = parts
      .map((p: any) => (typeof p.text === 'string' ? p.text : ''))
      .join('')
      .trim();
    if (allText) return allText;
  }

  if (typeof response.text === 'string' && response.text) {
    return response.text.trim();
  }

  if (typeof response.text === 'function') {
    try {
      const t = response.text();
      if (t) return String(t).trim();
    } catch {
      // ignore
    }
  }

  return '';
}

/**
 * Gọi trực tiếp REST API của Google Gemini bằng native fetch (vũ khí dự phòng 100% tin cậy trên trình duyệt)
 */
async function callGeminiREST(
  apiKey: string,
  modelId: string,
  options: {
    prompt: string;
    systemInstruction?: string;
    responseMimeType?: string;
    responseSchema?: any;
    maxOutputTokens?: number;
    temperature?: number;
  }
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${encodeURIComponent(apiKey)}`;

  const body: any = {
    contents: [
      {
        role: 'user',
        parts: [{ text: options.prompt }]
      }
    ],
    generationConfig: {
      temperature: options.temperature !== undefined ? options.temperature : 0.7,
      maxOutputTokens: options.maxOutputTokens || 8192
    }
  };

  if (options.systemInstruction) {
    body.systemInstruction = {
      parts: [{ text: options.systemInstruction }]
    };
  }

  if (options.responseMimeType) {
    body.generationConfig.responseMimeType = options.responseMimeType;
  }

  if (options.responseSchema) {
    body.generationConfig.responseSchema = options.responseSchema;
  }

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const errText = await res.text();
    let parsedErr: any;
    try {
      parsedErr = JSON.parse(errText);
    } catch {
      // ignore
    }
    const msg = parsedErr?.error?.message || `Lỗi HTTP ${res.status}: ${res.statusText}`;
    throw new Error(msg);
  }

  const data = await res.json();
  const text = extractResponseText(data);
  if (!text) {
    throw new Error('API trả về kết quả rỗng');
  }
  return text;
}

/**
 * Gọi Gemini API với cơ chế tự động thử lại (Retry) và Fallback tuần tự:
 * Thử SDK trước -> nếu lỗi thử REST API -> nếu lỗi chuyển sang model tiếp theo trong chuỗi.
 */
export async function callGeminiWithFallback(options: CallAIOptions): Promise<{ text: string; modelUsed: string }> {
  const apiKey = getStoredApiKey();
  if (!apiKey) {
    throw new Error('NO_API_KEY');
  }

  const preferredModel = getStoredModel();
  const candidateModels = [
    preferredModel,
    ...FALLBACK_CHAIN.filter((m) => m !== preferredModel)
  ];

  let lastError: unknown = null;

  for (const modelId of candidateModels) {
    try {
      if (options.onModelAttempt) {
        options.onModelAttempt(modelId);
      }

      // 1. Thử qua SDK GoogleGenAI
      try {
        const ai = new GoogleGenAI({ apiKey });
        const config: any = {
          maxOutputTokens: 8192,
          temperature: 0.7,
          systemInstruction:
            options.systemInstruction ||
            'Bạn là trợ lý AI Gia Sư Môn Toán THPT chuẩn sách Kết Nối Tri Thức. Bạn giải thích chi tiết, sư phạm, kèm công thức LaTeX ($...$, $$...$$), chỉ ra các bẫy thường gặp và hướng dẫn mẹo máy tính Casio fx-580VN X / 880BTG khi phù hợp.'
        };
        if (options.responseMimeType) {
          config.responseMimeType = options.responseMimeType;
        }

        const response = await ai.models.generateContent({
          model: modelId,
          contents: options.prompt,
          config
        });

        const responseText = extractResponseText(response);
        if (responseText && responseText.trim().length > 0) {
          return { text: responseText, modelUsed: modelId };
        }
      } catch (sdkErr: any) {
        console.warn(`SDK thất bại với model ${modelId}, chuyển sang REST API fallback:`, sdkErr);
      }

      // 2. Dự phòng: Thử qua REST API trực tiếp
      const restText = await callGeminiREST(apiKey, modelId, {
        prompt: options.prompt,
        systemInstruction: options.systemInstruction,
        responseMimeType: options.responseMimeType,
        responseSchema: options.responseSchema,
        maxOutputTokens: 8192,
        temperature: 0.7
      });

      if (restText && restText.trim().length > 0) {
        return { text: restText, modelUsed: modelId };
      }
    } catch (err: unknown) {
      console.warn(`Model ${modelId} thất bại:`, err);
      lastError = err;
    }
  }

  let errMsg = 'Không thể kết nối đến Gemini API';
  if (lastError instanceof Error) {
    errMsg = lastError.message;
  } else if (typeof lastError === 'object' && lastError !== null) {
    errMsg = (lastError as any).message || JSON.stringify(lastError);
  } else if (lastError) {
    errMsg = String(lastError);
  }

  throw new Error(`Tất cả các model AI đều thất bại: ${errMsg}`);
}

/**
 * Nhờ AI giải thích chi tiết câu hỏi toán học
 */
export async function askAIExplainQuestion(
  q: Question,
  userAnswer?: string,
  onModelAttempt?: (modelName: string) => void
): Promise<{ text: string; modelUsed: string }> {
  let questionDetails = `ĐỀ BÀI: ${q.content}\n`;
  if (q.format === 'multiple_choice_4' && q.options) {
    questionDetails += 'CÁC LỰA CHỌN:\n';
    q.options.forEach((opt) => {
      questionDetails += `- [${opt.id}] ${opt.text}\n`;
    });
    questionDetails += `ĐÁP ÁN ĐÚNG: ${q.correctAnswer}\n`;
    if (userAnswer) {
      questionDetails += `HỌC SINH ĐÃ CHỌN: ${userAnswer} (${userAnswer === q.correctAnswer ? 'Đúng' : 'Chưa đúng'})\n`;
    }
  } else if (q.format === 'true_false_4' && q.subQuestions) {
    questionDetails += 'CÁC MỆNH ĐỀ:\n';
    q.subQuestions.forEach((sub) => {
      questionDetails += `- [${sub.id}] ${sub.statement} (Đáp án: ${sub.isCorrect ? 'ĐÚNG' : 'SAI'})\n`;
    });
  } else if (q.format === 'short_answer') {
    questionDetails += `ĐÁP ÁN ĐÚNG: ${q.correctAnswer}\n`;
    if (userAnswer) {
      questionDetails += `HỌC SINH ĐÃ TRẢ LỜI: ${userAnswer}\n`;
    }
  }

  if (q.solution) {
    questionDetails += `HƯỚNG DẪN GIẢI HIỆN CÓ: ${q.solution}\n`;
  }

  const prompt = `
Dưới đây là một câu hỏi môn Toán THPT theo chương trình Kết Nối Tri Thức:
${questionDetails}

Hãy đóng vai Thầy/Cô giáo dạy Toán ôn thi tốt nghiệp THPT, phân tích giúp học sinh:
1. **Phương pháp tư duy & Lời giải chi tiết**: Trình bày từng bước mạch lạc, dùng chuẩn công thức LaTeX ($...$ hoặc $$...$$) đặc biệt là vectơ (\\overrightarrow{AB}, \\vec{a}), bảng biến thiên nếu có.
2. **Cảnh báo lỗi sai / Bẫy đề thi**: Học sinh thường mắc sai lầm gì ở bài này?
3. **Mẹo Casio (nếu có)**: Hướng dẫn bấm máy tính Casio fx-580VN X hoặc fx-880BTG để kiểm tra nhanh.
4. **Lời khuyên ôn tập**: Khắc sâu kiến thức trọng tâm của dạng toán này.
`;

  return callGeminiWithFallback({
    prompt,
    onModelAttempt
  });
}

/**
 * Nhờ AI tạo một câu hỏi toán học tương tự cùng dạng
 */
export async function askAIGenerateSimilarQuestion(
  q: Question,
  onModelAttempt?: (modelName: string) => void
): Promise<{ text: string; modelUsed: string }> {
  const prompt = `
Hãy tạo 1 câu hỏi môn Toán THPT TƯƠNG TỰ VÀ CÙNG DẠNG với câu hỏi sau nhưng thay đổi số liệu, tình huống:
ĐỀ BÀI GỐC: ${q.content}
DẠNG: ${q.format} (Mức độ: ${q.difficulty})

Yêu cầu xuất ra cấu trúc rõ ràng:
- **Đề bài tương tự**: (dùng LaTeX $...$ cho công thức)
- **Các phương án** (nếu là trắc nghiệm) hoặc **Các mệnh đề Đúng/Sai**
- **Đáp án chính xác**
- **Lời giải tóm tắt**
`;

  return callGeminiWithFallback({
    prompt,
    onModelAttempt
  });
}

export interface AIExamConfig {
  grade: GradeNumber;
  chapterId: string;
  chapterName: string;
  lessonName?: string;
  durationMinutes: number;
  difficulty: 'Nhận biết' | 'Thông hiểu' | 'Vận dụng' | 'Vận dụng cao' | 'Hỗn hợp phân hóa (Đề thi chuẩn)';
  counts: {
    part1Count: number;
    part2Count: number;
    part3Count: number;
  };
  customPrompt?: string;
}

export interface GeneratedExam {
  id: string;
  title: string;
  subtitle: string;
  grade: GradeNumber;
  chapterId: string;
  lessonName?: string;
  durationMinutes: number;
  createdAt: string;
  questions: ExamQuestion[];
}

/**
 * Chuẩn hóa các phương án trắc nghiệm A, B, C, D
 */
function normalizeOptions(rawOptions: any): { id: string; text: string }[] {
  if (!rawOptions) return [];
  if (Array.isArray(rawOptions)) {
    return rawOptions.map((opt, idx) => {
      const defaultId = ['A', 'B', 'C', 'D'][idx] || `opt_${idx + 1}`;
      if (typeof opt === 'string') {
        const match = opt.match(/^([A-D])[\.\:\)]\s*(.*)$/i);
        if (match) {
          return { id: match[1].toUpperCase(), text: match[2].trim() };
        }
        return { id: defaultId, text: opt.trim() };
      }
      if (typeof opt === 'object' && opt !== null) {
        return {
          id: String(opt.id || opt.key || defaultId).toUpperCase().trim(),
          text: String(opt.text || opt.content || opt.value || '').trim()
        };
      }
      return { id: defaultId, text: String(opt).trim() };
    });
  }
  if (typeof rawOptions === 'object') {
    return Object.entries(rawOptions).map(([key, value]) => ({
      id: key.toUpperCase().trim(),
      text: typeof value === 'string' ? value.trim() : String(value)
    }));
  }
  return [];
}

/**
 * Chuẩn hóa các mệnh đề Đúng/Sai a, b, c, d
 */
function normalizeSubQuestions(rawSubs: any): { id: string; statement: string; isCorrect: boolean }[] {
  if (!rawSubs) return [];
  if (Array.isArray(rawSubs)) {
    return rawSubs.map((sub, idx) => {
      const defaultId = ['a', 'b', 'c', 'd'][idx] || `sub_${idx + 1}`;
      if (typeof sub === 'object' && sub !== null) {
        const id = String(sub.id || sub.key || defaultId).toLowerCase().trim();
        const statement = String(sub.statement || sub.content || sub.text || '').trim();
        const isCorrect =
          sub.isCorrect === true ||
          sub.correct === true ||
          sub.answer === true ||
          String(sub.isCorrect).toLowerCase() === 'true' ||
          String(sub.answer).toLowerCase() === 'đúng' ||
          String(sub.answer).toLowerCase() === 'true';
        return { id, statement, isCorrect };
      }
      return { id: defaultId, statement: String(sub).trim(), isCorrect: false };
    });
  }
  return [];
}

/**
 * Bóc tách an toàn vỏ bọc markdown ```json ... ``` ở ngoài cùng
 */
function stripOuterMarkdownCodeBlock(raw: string): string {
  let s = raw.trim();

  // Bỏ ```json hoặc ``` ở đầu
  if (s.startsWith('```')) {
    const firstNewline = s.indexOf('\n');
    if (firstNewline !== -1) {
      s = s.substring(firstNewline + 1);
    } else {
      s = s.replace(/^```[a-zA-Z]*\s*/, '');
    }

    // Bỏ ``` ở cuối nếu có
    const lastFence = s.lastIndexOf('```');
    if (lastFence !== -1) {
      s = s.substring(0, lastFence);
    } else if (s.endsWith('```')) {
      s = s.substring(0, s.length - 3);
    }
  }

  // Cắt từ dấu { đầu tiên
  const firstBrace = s.indexOf('{');
  if (firstBrace !== -1) {
    s = s.substring(firstBrace);
  }

  return s.trim();
}

/**
 * Bộ tiền xử lý và sửa lỗi JSON siêu bền bỉ (Bulletproof JSON Repair):
 * - Xử lý công thức LaTeX (\frac, \sqrt, \vec, \alpha, \text, \times, \tan,...) để không vỡ JSON
 * - Tự động escape các ký tự điều khiển (raw newlines, tabs)
 * - Tự động sửa các dấu nháy kép nội dung chưa escape (ví dụ: "cạnh "a" của hình")
 * - Tự động loại bỏ trailing commas (, } và , ])
 * - Tự động đóng các cặp ngoặc khi chuỗi JSON bị ngắt giữa chừng do chạm giới hạn token
 */
export function sanitizeAndRepairJSON(raw: string): string {
  if (!raw || !raw.trim()) {
    throw new Error('Dữ liệu AI trả về rỗng.');
  }

  let text = stripOuterMarkdownCodeBlock(raw);

  const firstBrace = text.indexOf('{');
  if (firstBrace === -1) {
    throw new Error('Không tìm thấy khối JSON trong dữ liệu phản hồi.');
  }
  text = text.substring(firstBrace);

  let result = '';
  let inString = false;
  let isEscaped = false;
  const braceStack: ('{' | '[')[] = [];

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const nextCh = text[i + 1] || '';

    if (!inString) {
      if (ch === '"') {
        inString = true;
        result += ch;
      } else if (ch === '{' || ch === '[') {
        braceStack.push(ch);
        result += ch;
      } else if (ch === '}') {
        if (braceStack.length > 0 && braceStack[braceStack.length - 1] === '{') {
          braceStack.pop();
        }
        result += ch;
        // Nếu đã đóng hết tất cả ngoặc của root object, dừng lại để bỏ qua lời chat thừa phía sau
        if (braceStack.length === 0) {
          break;
        }
      } else if (ch === ']') {
        if (braceStack.length > 0 && braceStack[braceStack.length - 1] === '[') {
          braceStack.pop();
        }
        result += ch;
      } else {
        result += ch;
      }
    } else {
      // Đang ở TRONG chuỗi ký tự "..."
      if (isEscaped) {
        isEscaped = false;
        result += ch;
      } else if (ch === '\\') {
        // Kiểm tra các ký tự theo sau dấu gạch chéo
        const nextLetters = text.substring(i + 1, i + 8).toLowerCase();
        const isLatexCommand = /^(text|times|tan|tau|theta|tilde|right|rho|beta|bar|begin|bullet|frac|dfrac|neq|nabla|nu|notin|left|sqrt|vec|over|alpha|gamma|delta|pi|in|sum|int|lim|le|ge|cdot|pm|cos|sin|cot|log|ln|subset|cup|cap|infty)/.test(
          nextLetters
        );

        if (isLatexCommand) {
          // Lệnh LaTeX bắt đầu bằng ký tự điều khiển -> nhân đôi gạch chéo để JSON.parse hiểu là dấu \
          result += '\\\\';
        } else if (nextCh === '"' || nextCh === '\\' || nextCh === '/') {
          result += ch;
          isEscaped = true;
        } else if (
          nextCh === 'u' &&
          /^[0-9a-fA-F]{4}$/.test(text.substring(i + 2, i + 6))
        ) {
          result += ch;
          isEscaped = true;
        } else if (nextCh === 'b' || nextCh === 'f' || nextCh === 'n' || nextCh === 'r' || nextCh === 't') {
          result += ch;
          isEscaped = true;
        } else {
          // Tất cả các ký tự thoát khác (như \alpha, \vec, \Delta, \int,...) -> nhân đôi \
          result += '\\\\';
        }
      } else if (ch === '"') {
        // Kiểm tra xem đây có phải là dấu đóng string JSON thực sự hay dấu nháy trong nội dung văn bản:
        // Dấu đóng string JSON hợp lệ theo sau bởi: khoảng trắng rồi đến ',', '}', ']', hoặc ':'
        const remaining = text.substring(i + 1, i + 25).trimStart();
        const isLikelyStringClosing = remaining.length === 0 || /^[,\}\]\:]/.test(remaining);

        if (isLikelyStringClosing) {
          inString = false;
          result += ch;
        } else {
          // Dấu nháy kép nội dung (ví dụ: "cạnh "a"") -> tự động escape thành \"
          result += '\\"';
        }
      } else if (ch === '\n') {
        result += '\\n';
      } else if (ch === '\r') {
        result += '\\r';
      } else if (ch === '\t') {
        result += '\\t';
      } else {
        result += ch;
      }
    }
  }

  // Phục hồi chuỗi JSON bị ngắt giữa chừng do chạm giới hạn token
  if (inString) {
    result += '"';
  }

  // Xóa trailing commas: ", }" -> "}" và ", ]" -> "]"
  result = result.replace(/,\s*([\}\]])/g, '$1');

  // Xóa thuộc tính dở dang ở cuối trước khi đóng ngoặc
  result = result.replace(/,\s*"[^"]*"?\s*:\s*$/g, '');
  result = result.replace(/,\s*"[^"]*"?\s*$/g, '');
  result = result.replace(/,\s*$/g, '');

  // Đóng tất cả các ngoặc mở còn thiếu theo thứ tự LIFO
  while (braceStack.length > 0) {
    const unclosed = braceStack.pop();
    if (unclosed === '{') {
      result += '}';
    } else if (unclosed === '[') {
      result += ']';
    }
  }

  return result;
}

/**
 * Cứu nguy tối thượng: Trích xuất danh sách câu hỏi độc lập bằng Regex
 * khi toàn bộ JSON root bị vỡ cú pháp.
 */
function extractQuestionsFallback(raw: string): any[] {
  const questions: any[] = [];
  // Regex tìm từng block câu hỏi độc lập có "content"
  const qRegex = /\{[^{}]*?"content"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*?\}/g;

  let match: RegExpExecArray | null;
  while ((match = qRegex.exec(raw)) !== null) {
    try {
      const qBlock = match[0];
      const repaired = sanitizeAndRepairJSON(qBlock);
      const parsedQ = JSON.parse(repaired);
      if (parsedQ && parsedQ.content) {
        questions.push(parsedQ);
      }
    } catch {
      // ignore
    }
  }

  return questions;
}

/**
 * Trích xuất và phân tích JSON từ AI một cách bền bỉ (multi-strategy resilient parser)
 */
export function parseJSONFromLLM(text: string): any {
  if (!text || !text.trim()) {
    throw new Error('Dữ liệu AI trả về rỗng.');
  }

  const clean = stripOuterMarkdownCodeBlock(text);

  // Lần thử 1: Parse trực tiếp
  try {
    const parsed = JSON.parse(clean);
    if (parsed && (Array.isArray(parsed.questions) || parsed.title)) {
      return parsed;
    }
  } catch {
    // Chuyển sang Lần thử 2
  }

  // Lần thử 2: Chạy bộ phục hồi LaTeX / Unescaped Control Chars / Truncated JSON
  try {
    const repaired = sanitizeAndRepairJSON(clean);
    const parsed = JSON.parse(repaired);
    if (parsed && (Array.isArray(parsed.questions) || parsed.title)) {
      return parsed;
    }
  } catch {
    // Chuyển sang Lần thử 3
  }

  // Lần thử 3: Tìm mảng "questions"
  try {
    const questionMatch = clean.match(/"questions"\s*:\s*(\[[\s\S]*?\])/);
    if (questionMatch) {
      const repairedArray = sanitizeAndRepairJSON(questionMatch[1]);
      const questions = JSON.parse(repairedArray);
      if (Array.isArray(questions) && questions.length > 0) {
        return {
          title: 'Đề Kiểm Tra Môn Toán THPT',
          subtitle: 'Biên soạn bởi AI KNTT',
          questions
        };
      }
    }
  } catch {
    // Chuyển sang Lần thử 4
  }

  // Lần thử 4: Cứu nguy tối thượng bằng Regex Question Extractor
  const fallbackQuestions = extractQuestionsFallback(clean);
  if (fallbackQuestions.length > 0) {
    return {
      title: 'Đề Kiểm Tra Môn Toán THPT',
      subtitle: 'Biên soạn bởi AI KNTT',
      questions: fallbackQuestions
    };
  }

  throw new Error('Không thể phân tích dữ liệu đề thi do AI trả về định dạng JSON chưa hoàn chỉnh. Vui lòng bấm tạo lại!');
}

/**
 * JSON Schema định dạng chuẩn cho đề thi Toán (Structured Outputs)
 * Ép buộc Gemini API luôn trả về JSON hợp lệ 100%.
 */
const EXAM_JSON_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    subtitle: { type: 'string' },
    questions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          part: { type: 'string', enum: ['I', 'II', 'III'] },
          format: { type: 'string', enum: ['multiple_choice_4', 'true_false_4', 'short_answer'] },
          difficulty: { type: 'string' },
          content: { type: 'string' },
          options: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'string' },
                text: { type: 'string' }
              },
              required: ['id', 'text']
            }
          },
          subQuestions: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'string' },
                statement: { type: 'string' },
                isCorrect: { type: 'boolean' }
              },
              required: ['id', 'statement', 'isCorrect']
            }
          },
          correctAnswer: { type: 'string' },
          solution: { type: 'string' }
        },
        required: ['id', 'part', 'format', 'difficulty', 'content', 'solution']
      }
    }
  },
  required: ['title', 'subtitle', 'questions']
};

/**
 * Tự động tạo đề thi môn Toán theo yêu cầu bằng AI Gemini
 * Tích hợp đa tầng: SDK GoogleGenAI + REST API Native Fetch + Structured Outputs JSON Schema
 * + Bộ phục hồi JSON chống lỗi LaTeX + Tự động Fallback chuỗi model.
 */
export async function generateAIExam(
  config: AIExamConfig,
  onModelAttempt?: (modelName: string) => void
): Promise<{ exam: GeneratedExam; modelUsed: string }> {
  const apiKey = getStoredApiKey();
  if (!apiKey) {
    throw new Error('Vui lòng nhập Gemini API Key để bắt đầu tạo đề!');
  }

  const lessonScope =
    config.lessonName && config.lessonName !== 'all'
      ? `- Trọng tâm bài học cụ thể: "${config.lessonName}"`
      : `- Phạm vi: Toàn bộ các bài học trong chương "${config.chapterName}"`;

  const prompt = `
Bạn là chuyên gia thẩm định và ra đề thi môn Toán THPT theo chương trình GDPT mới (Sách giáo khoa Kết Nối Tri Thức Với Cuộc Sống), chuẩn ma trận định dạng kỳ thi Tốt nghiệp THPT từ năm 2026 trở đi.
Hãy biên soạn một bộ đề thi kiểm tra theo đúng ma trận và yêu cầu sau:

THÔNG TIN BẮT BUỘC:
- Khối lớp: Lớp ${config.grade}
- Chương: ${config.chapterName}
${lessonScope}
- Thời gian làm bài: ${config.durationMinutes} phút
- Mức độ nhận thức: ${config.difficulty}
- Cơ cấu câu hỏi BẮT BUỘC:
  + Phần I (Trắc nghiệm 4 lựa chọn A, B, C, D; chỉ 1 đáp án đúng): đúng ${config.counts.part1Count} câu
  + Phần II (Trắc nghiệm Đúng / Sai, mỗi câu có đúng 4 ý a, b, c, d): đúng ${config.counts.part2Count} câu
  + Phần III (Trả lời ngắn / Điền kết quả số): đúng ${config.counts.part3Count} câu
${config.customPrompt ? `- Yêu cầu bổ sung đặc biệt: ${config.customPrompt}` : ''}

QUY TẮC TIẾNG VIỆT & TOÁN HỌC BẮT BUỘC:
1. Toàn bộ câu hỏi, phương án, mệnh đề và lời giải PHẢI viết bằng tiếng Việt chuẩn ngữ pháp Bộ Giáo dục & Đào tạo, có dấu đầy đủ và chuẩn xác.
2. Tất cả công thức toán, biến số, tọa độ, nửa khoảng đều phải đặt trong dấu LaTeX $...$ (inline) hoặc $$...$$ (display). Ví dụ: $\\frac{a}{b}$, $\\vec{a}$, $\\overrightarrow{AB}$, $\\sqrt{x}$, $\\alpha$, $[a; b)$.
3. QUAN TRỌNG VỀ ĐỊNH DẠNG JSON: Trong chuỗi JSON, viết dấu gạch chéo ngược kép cho LaTeX (\\\\frac, \\\\sqrt, \\\\vec) để đảm bảo chuỗi JSON hợp lệ. Tuyệt đối KHÔNG để dấu nháy kép trần chưa escape trong chuỗi text (nếu cần trích dẫn hãy dùng dấu nháy đơn '...' hoặc ngoặc kép cong “...”).
4. Phần I: Trường "options" phải có đủ 4 phần tử với id: 'A', 'B', 'C', 'D' và "correctAnswer" là một trong 4 chữ cái 'A', 'B', 'C', 'D'.
5. Phần II: Trường "subQuestions" phải có đủ 4 phần tử với id: 'a', 'b', 'c', 'd' và statement, isCorrect (true hoặc false).
6. Phần III: "correctAnswer" phải là một số nguyên hoặc số thập phân/phân số gọn (ví dụ: '12', '4.5', '3/4').
7. Mỗi câu phải có trường "solution" giải thích chi tiết từng bước bằng tiếng Việt sư phạm chuẩn mực.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ trả về DUY NHẤT một chuỗi JSON hợp lệ tuân theo schema:
{
  "title": "Đề Kiểm Tra Toán ${config.grade} - ${config.lessonName && config.lessonName !== 'all' ? config.lessonName : config.chapterName} (Chuẩn 2026 Trở Đi)",
  "subtitle": "Thời gian ${config.durationMinutes} phút • Khung chương trình Kết Nối Tri Thức",
  "questions": [
    {
      "id": "q1",
      "part": "I",
      "format": "multiple_choice_4",
      "difficulty": "Thông hiểu",
      "content": "Nội dung câu hỏi...",
      "options": [
        {"id": "A", "text": "Phương án A"},
        {"id": "B", "text": "Phương án B"},
        {"id": "C", "text": "Phương án C"},
        {"id": "D", "text": "Phương án D"}
      ],
      "correctAnswer": "A",
      "solution": "Lời giải chi tiết..."
    },
    {
      "id": "q2",
      "part": "II",
      "format": "true_false_4",
      "difficulty": "Vận dụng",
      "content": "Nội dung câu hỏi đúng sai...",
      "subQuestions": [
        {"id": "a", "statement": "Mệnh đề a...", "isCorrect": true},
        {"id": "b", "statement": "Mệnh đề b...", "isCorrect": false},
        {"id": "c", "statement": "Mệnh đề c...", "isCorrect": true},
        {"id": "d", "statement": "Mệnh đề d...", "isCorrect": false}
      ],
      "solution": "Lời giải chi tiết..."
    },
    {
      "id": "q3",
      "part": "III",
      "format": "short_answer",
      "difficulty": "Vận dụng",
      "content": "Nội dung câu hỏi điền số...",
      "correctAnswer": "12",
      "solution": "Lời giải chi tiết..."
    }
  ]
}
`;

  const preferredModel = getStoredModel();
  const candidateModels = [
    preferredModel,
    ...FALLBACK_CHAIN.filter((m) => m !== preferredModel)
  ];

  let lastError: unknown = null;

  for (const modelId of candidateModels) {
    try {
      if (onModelAttempt) {
        onModelAttempt(modelId);
      }

      let responseText = '';

      // Chiến lược 1: Gọi qua SDK GoogleGenAI với responseSchema
      try {
        const ai = new GoogleGenAI({ apiKey });
        const configObj: any = {
          systemInstruction:
            'Bạn là chuyên gia soạn thảo đề thi môn Toán THPT Kết Nối Tri Thức. Bạn CHỈ trả về dữ liệu định dạng JSON chuẩn không bọc thêm bất kỳ lời nói chuyện nào.',
          responseMimeType: 'application/json',
          responseSchema: EXAM_JSON_SCHEMA,
          maxOutputTokens: 8192,
          temperature: 0.7
        };

        const response = await ai.models.generateContent({
          model: modelId,
          contents: prompt,
          config: configObj
        });

        responseText = extractResponseText(response);
      } catch (sdkErr: any) {
        console.warn(`SDK gặp lỗi với model ${modelId}, chuyển sang gọi REST API native:`, sdkErr);
      }

      // Chiến lược 2: Nếu SDK không trả về kết quả, dùng REST API native fetch (kèm responseSchema)
      if (!responseText || responseText.trim().length === 0) {
        try {
          responseText = await callGeminiREST(apiKey, modelId, {
            prompt,
            systemInstruction:
              'Bạn là chuyên gia soạn thảo đề thi môn Toán THPT Kết Nối Tri Thức. Bạn CHỈ trả về dữ liệu định dạng JSON chuẩn.',
            responseMimeType: 'application/json',
            responseSchema: EXAM_JSON_SCHEMA,
            maxOutputTokens: 8192,
            temperature: 0.7
          });
        } catch (restErrWithSchema: any) {
          // Chiến lược 3: Nếu model không nhận responseSchema, thử gọi REST không có schema
          console.warn(`Model ${modelId} không hỗ trợ schema, thử lại không có schema:`, restErrWithSchema);
          responseText = await callGeminiREST(apiKey, modelId, {
            prompt,
            systemInstruction:
              'Bạn là chuyên gia soạn thảo đề thi môn Toán THPT Kết Nối Tri Thức. Bạn CHỈ trả về dữ liệu định dạng JSON chuẩn.',
            responseMimeType: 'application/json',
            maxOutputTokens: 8192,
            temperature: 0.7
          });
        }
      }

      if (!responseText || responseText.trim().length === 0) {
        throw new Error(`Model ${modelId} trả về kết quả rỗng.`);
      }

      // Phân tích và sửa lỗi JSON đa tầng
      const parsed = parseJSONFromLLM(responseText);

      if (!parsed || !Array.isArray(parsed.questions) || parsed.questions.length === 0) {
        throw new Error(`Model ${modelId} không tạo được danh sách câu hỏi hợp lệ.`);
      }

      const generatedExam: GeneratedExam = {
        id: `ai_exam_${Date.now()}`,
        title: parsed.title || `Đề Kiểm Tra Toán ${config.grade} - ${config.chapterName}`,
        subtitle: parsed.subtitle || `Thời gian: ${config.durationMinutes} phút • Soạn bởi AI KNTT`,
        grade: config.grade,
        chapterId: config.chapterId,
        lessonName: config.lessonName,
        durationMinutes: config.durationMinutes,
        createdAt: new Date().toLocaleString('vi-VN'),
        questions: (parsed.questions || []).map((q: any, idx: number) => {
          const rawPart = q.part ? String(q.part).toUpperCase().trim() : '';
          const rawFormat = q.format || '';
          const format =
            rawFormat === 'true_false_4' || rawPart === 'II'
              ? 'true_false_4'
              : rawFormat === 'short_answer' || rawPart === 'III'
              ? 'short_answer'
              : 'multiple_choice_4';
          const part =
            rawPart === 'II' || format === 'true_false_4'
              ? 'II'
              : rawPart === 'III' || format === 'short_answer'
              ? 'III'
              : 'I';

          return {
            id: q.id || `ai_q_${idx + 1}`,
            part,
            format,
            difficulty: q.difficulty || config.difficulty || 'Thông hiểu',
            content: q.content || `Câu hỏi ${idx + 1}`,
            options: format === 'multiple_choice_4' ? normalizeOptions(q.options) : undefined,
            subQuestions: format === 'true_false_4' ? normalizeSubQuestions(q.subQuestions) : undefined,
            correctAnswer:
              q.correctAnswer !== undefined && q.correctAnswer !== null ? String(q.correctAnswer).trim() : '',
            solution: q.solution || ''
          };
        })
      };

      return {
        exam: generatedExam,
        modelUsed: modelId
      };
    } catch (err: unknown) {
      console.warn(`Tạo đề với model ${modelId} thất bại:`, err);
      lastError = err;
      // Tự động thử lại với model tiếp theo trong Fallback chain
    }
  }

  let errMsg = 'Không thể kết nối đến Gemini API';
  if (lastError instanceof Error) {
    errMsg = lastError.message;
  } else if (typeof lastError === 'object' && lastError !== null) {
    errMsg = (lastError as any).message || JSON.stringify(lastError);
  } else if (lastError) {
    errMsg = String(lastError);
  }

  throw new Error(`Tất cả các model AI đều thất bại: ${errMsg}`);
}
