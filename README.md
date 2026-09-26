# 知向教育：个性化 AI 解题 Demo

一个可运行的 Streamlit MVP：先用 10 道核心题生成 Schwartz 十维价值向量，再把向量转成明确的教育策略与结构化 Prompt，调用 DeepSeek 生成类似“作业帮”的分步解答。

## 已实现

- 浅色、紫色强调、Step 1/2/3 卡片式界面
- Schwartz 10 维简化问卷与 Value Vector
- Value Vector → 教学策略 → 结构化 Prompt
- 文字题目录入、题目图片上传与可替换图片适配层
- 知识点、思路、分步讲解、最终答案、举一反三
- 基于原题和上下文继续追问
- 本机保存、JSON 导入/导出用户画像与聊天历史
- 无 API Key 时可体验明确标注的离线流程，不伪造答案

## 快速运行

建议使用 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

浏览器会打开 `http://localhost:8501`。

## 配置 DeepSeek

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Key 只从环境变量或本机 `.env` 读取，源码中没有写死。模型名称可能随 DeepSeek 服务更新，请以控制台当前可用模型为准。

## 图片识别适配层

默认配置为：

```dotenv
VISION_PROVIDER=disabled
```

此时图片可以上传和预览，界面会提示学生补充文字，不会假装已经识别。

如果账号与所选 DeepSeek 模型支持图片输入，可配置：

```dotenv
VISION_PROVIDER=deepseek_vision
DEEPSEEK_VISION_MODEL=deepseek-v4-flash
```

适配层位于 `src/vision.py`。之后可以新增 PaddleOCR、云 OCR 或其他多模态模型，只需实现相同的 `extract(image_bytes, mime_type)` 接口，不需要改动问卷、聊天或页面流程。

## 数据保存

- 当前画像、题目文字和对话保存在 `data/state.json`。
- 上传的原始图片不会保存。
- 侧边栏支持导出和重新导入 JSON。
- 这是单机 MVP；多人上线时应改为登录用户隔离的数据库，并增加加密、保留期限与删除机制。

## 项目结构

```text
.
├── app.py                       # Streamlit 页面与三步流程
├── src/
│   ├── questionnaire.py        # 10 维问卷与计分
│   ├── personalization.py      # Value → 教学策略 → Prompt
│   ├── deepseek_client.py      # DeepSeek API 与离线演示
│   ├── vision.py               # OCR/多模态可替换适配层
│   └── storage.py              # 本地保存与导出
├── tests/test_core.py
├── requirements.txt
└── .env.example
```

## 验证

```bash
python -m pytest -q
python -m compileall app.py src
```

## 下一步升级

### 1. 自适应问卷

第一层保留 10 道核心题；第二层只在以下情况动态追加：Top 价值缺少验证、Schwartz 对立方向同时高分、向量过于平坦或缺少明显高值。加入冲突题与置信度后，把平均题量控制在约 13–18 题。

### 2. A/B 实验

随机分为三组：普通 Prompt、直接输入 Value Vector、结构化个性化策略。比较正确率、价值匹配度、解释满意度、继续追问率与完成时间；正确率必须作为硬约束，避免“更个性化但更不准确”。

### 3. OpenAI 版本

抽象统一的 `TutorProvider` 接口，在保留 `build_system_prompt()` 的前提下新增 OpenAI provider。图片题可以接入支持视觉输入的模型；实验中使用相同题集和相同输出结构，减少模型差异带来的混杂。

### 4. 产品化

增加登录与多用户数据隔离、学科分类器、公式 OCR、教师审核、答案引用与反馈标注。价值画像应允许查看、修正和删除，并明确它只调整教学表达，不用于能力评定。

