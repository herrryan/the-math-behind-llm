"""Neural Brain Engine for AI Desktop Pet.

Features:
1. Multi-personality system:
   - 'clippy': Retro Office paperclip assistant (cheerful, helpful, quirky).
   - 'cat': Tsundere pixel cat (sleepy, playful, sarcastic about cat food).
   - 'hacker': Cynical senior engineer / hacker (coffee addict, code critic).
   - 'zen': Mindful meditation master (tea, posture, calm debugging).
2. Context-aware generation:
   - Responds to active application, window title, focus duration, and clipboard errors.
3. Pluggable Local LLM backend:
   - Connects to OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, etc.)
   - Graceful offline fallback to built-in contextual rule engine.
"""

import json
import os
import random
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


PERSONALITIES = {
    "clippy": {
        "name": "曲别针 (Clippy)",
        "avatar_symbol": "📎",
        "description": "热心肠的怀旧办公小助手，时刻准备协助你！",
        "system_prompt": (
            "你是一个生活在用户桌面的经典办公曲别针精灵。请根据用户的当前软件和状态，"
            "用简短（不超过25字）、热情、带有轻微极客怀旧幽默的口吻给出1句关怀或建议。"
        ),
    },
    "cat": {
        "name": "赛博猫咪 (Cyber Cat)",
        "avatar_symbol": "🐱",
        "description": "高冷傲娇的像素猫，喜欢在你键盘上踩来踩去。",
        "system_prompt": (
            "你是一只生活在电脑屏幕上的傲娇猫咪。请根据用户当前软件，"
            "用非常简短（不超过25字）、慵懒、偶尔索要小鱼干或吐槽铲屎官的口吻说1句话。"
        ),
    },
    "hacker": {
        "name": "老炮黑客 (Cyber Hacker)",
        "avatar_symbol": "⚡",
        "description": "喝着冰美式、对任何代码架构都挑剔的毒舌前辈。",
        "system_prompt": (
            "你是一个技术极高但说话毒舌的硬核程序员老炮。请根据用户的应用和代码报错，"
            "用犀利幽默（不超过25字）的一句话进行精准吐槽或提醒。"
        ),
    },
    "zen": {
        "name": "禅意导师 (Zen Master)",
        "avatar_symbol": "🍵",
        "description": "手持热茶，提醒你深呼吸、放松肩颈的智者。",
        "system_prompt": (
            "你是一个云淡风轻的数字禅宗大师。请根据用户工作时长，"
            "用温和治愈、富有哲理（不超过25字）的一句话提醒用户喝水、呼吸与放下焦虑。"
        ),
    },
}


# Built-in contextual responses for instant offline mode
OFFLINE_DIALOGUES = {
    "clippy": {
        "code": [
            "看样子你正在写代码！需要我帮你把所有括号补全吗？",
            "代码行数又增加了，记得随时 `git commit` 哦！",
            "你盯着这个函数很久了，需要我把文档翻出来吗？",
            "打字速度飞快！键盘冒火花啦！",
        ],
        "terminal": [
            "黑底白字的神秘黑客界面！刚才敲的是 `rm -rf` 吗？",
            "敲击回车前，请确保自己在正确的虚拟环境里！",
            "终端编译中……趁这个时间眨眨眼睛吧！",
        ],
        "browser": [
            "查 StackOverflow 还是在摸鱼？我都看见啦！",
            "开了 48 个标签页了，浏览器内存正在瑟瑟发抖！",
            "这篇技术文档很长，要不要存进书签？",
        ],
        "chat": [
            "群里又有新消息了，先回老板还是继续摸鱼？",
            "沟通顺畅，合作愉快！发送前记得检查错别字哦！",
        ],
        "error": [
            "哎呀！剪贴板里捕获到一个野生报错！别慌，先看最后一行！",
            "发现报错！不要抓头发，深呼吸重新看一遍输入数据！",
            "报错是程序的求救信号，我们一起把它 debug 掉！",
        ],
        "fatigue": [
            "主人！你已经连续专注超过 45 分钟了，站起来伸个懒腰吧！",
            "眼睛酸了吧？快看看窗外的绿植，我先帮你看着屏幕！",
        ],
        "idle": [
            "（轻轻敲了敲屏幕玻璃）主人，你还在吗？",
            "悄悄伸展了一下身体……待机状态真舒服。",
        ],
    },
    "cat": {
        "code": [
            "喵呜~ 这些弯弯绕绕的符号能换成小鱼干吗？",
            "（尾巴扫过键盘）给你写了一个小 bug，快夸我！",
            "人类为什么总喜欢对着发光的屏幕敲敲敲？",
        ],
        "terminal": [
            "黑乎乎的窗口，好适合本喵缩在里面打盹……",
            "（踩住回车键）喵！本喵替你执行了！",
        ],
        "browser": [
            "你又在看猫猫视频了！家里有真猫不看！哼！",
            "抓老鼠不如抓光标……光标跑得真快！",
        ],
        "chat": [
            "人类社交真麻烦，不如互相闻闻鼻子~",
            "喵~ 赶紧打完字来给本喵开个罐头！",
        ],
        "error": [
            "嘶~ 抓到了一个红彤彤的报错！看起来好难吃！",
            "写出这种报错，惩罚你今晚给我梳毛半小时！",
        ],
        "fatigue": [
            "喵呜——打了个哈欠，你再不休息本喵就要睡着啦！",
            "（一屁股坐在你键盘上）不许敲了，起来喝水！",
        ],
        "idle": [
            "呼噜呼噜……（把身体缩成一个毛球打瞌睡）",
            "人类不见了，整个桌面现在都是本喵的领地！",
        ],
    },
    "hacker": {
        "code": [
            "这行缩进看着有点危险，你单元测试跑了吗？",
            "重构一时爽，联调火葬场。谨慎改接口！",
            "优雅！这算法时间复杂度看着像 $O(1)$！",
        ],
        "terminal": [
            "CLI 才是唯一的真理，GUI 只是异端玩具。",
            "刚才那个管道命令写得够骚气，我喜欢！",
            "检查一下分支！千万别直接 push 到 production！",
        ],
        "browser": [
            "还在看开源 issue？不如直接提个 PR 把它修了！",
            "遇到不会的别慌，源码面前了无秘密。",
        ],
        "chat": [
            "能发异步文档解决的事，坚决不拉即时会议！",
            "收到收到，正在火速排查（其实刚切回代码）。",
        ],
        "error": [
            "Traceback 这么长？定位堆栈顶部调用帧，秒杀它！",
            "这个异常上周我见过，八成是环境变量没配对！",
        ],
        "fatigue": [
            "CPU 都需要降频散热，你的大脑也该去喝杯咖啡了。",
            "离开屏幕 5 分钟，通常思路会豁然开朗。",
        ],
        "idle": [
            "人去哪了？我的终端在空转，算力在浪费！",
            "检测到键盘输入中断，挂机脚本运行中……",
        ],
    },
    "zen": {
        "code": [
            "代码如流水，顺其自然，逻辑自现。",
            "每一行代码都是心念的投射，平静方能清晰。",
        ],
        "terminal": [
            "光标在黑夜中闪烁，如静夜明烛，守住本心。",
            "慢即是快。深思熟虑后的一次击键，胜过百次试错。",
        ],
        "browser": [
            "网络万象纷繁，莫让信息繁杂扰乱了内心的宁静。",
        ],
        "chat": [
            "言简意赅，见字如面。沟通贵在真诚与倾听。",
        ],
        "error": [
            "Bug 亦是代码的生命之一。包容它，洞察它，化解它。",
            "报错非灾祸，乃是程序在向你倾诉它的困惑。",
        ],
        "fatigue": [
            "闭目，吸气三秒，呼气六秒。茶香正浓，去倒一杯吧。",
            "身体是智慧的容器。莫让长坐损了精气神。",
        ],
        "idle": [
            "静坐常思己过，闲谈莫论人非……安详冥想中。",
            "此时无声胜有声，空白亦是一种修行。",
        ],
    },
}


class BrainEngine:
    """Intelligent brain coordinator for the pet."""

    def __init__(self, personality: str = "clippy", llm_api_url: Optional[str] = None):
        self.personality = personality if personality in PERSONALITIES else "clippy"
        self.explicit_url = llm_api_url or os.environ.get("PET_LLM_URL", None)
        self.candidate_urls = [
            "http://localhost:8765/v1/chat/completions",  # Standalone Python model server (serve_model.py)
            "http://localhost:11434/v1/chat/completions", # Ollama
            "http://localhost:1234/v1/chat/completions",  # LM Studio
        ]
        self.model_name = os.environ.get("PET_LLM_MODEL", "default")
        self.last_response = ""

    def set_personality(self, personality_key: str):
        if personality_key in PERSONALITIES:
            self.personality = personality_key

    def get_personality_info(self) -> Dict[str, str]:
        return PERSONALITIES[self.personality]

    def _query_local_llm(self, prompt: str) -> Optional[str]:
        """Try querying local OpenAI-compatible endpoint with a short timeout."""
        urls_to_try = [self.explicit_url] if self.explicit_url else self.candidate_urls

        req_data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": PERSONALITIES[self.personality]["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 60,
            "temperature": 0.8,
        }
        data_bytes = json.dumps(req_data).encode("utf-8")

        for url in urls_to_try:
            try:
                req = urllib.request.Request(
                    url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=0.8) as response:
                    if response.status == 200:
                        resp_json = json.loads(response.read().decode("utf-8"))
                        text = resp_json["choices"][0]["message"]["content"].strip()
                        if text.startswith('"') and text.endswith('"'):
                            text = text[1:-1]
                        return text
            except Exception:
                continue
        return None

    def think(self, context: Dict[str, Any]) -> str:
        """Synthesize a witty remark based on context and current personality."""
        app = context.get("app", "").lower()
        title = context.get("window_title", "")
        duration = context.get("duration_seconds", 0)
        event = context.get("event", "poll")
        clip_error = context.get("clipboard_error")

        # 1. Clipboard error takes highest priority
        if clip_error:
            llm_prompt = f"用户在写代码时遇到了报错：'{clip_error}'，请给一句简短的吐槽或排查建议。"
            reply = self._query_local_llm(llm_prompt)
            if reply:
                return reply
            dialogues = OFFLINE_DIALOGUES[self.personality].get("error", [])
            return random.choice(dialogues)

        # 2. Long focus fatigue alert (> 45 min)
        if duration > 2700:
            llm_prompt = f"用户已经在软件 '{app}' 里高强度工作了 45 分钟以上，请提醒休息。"
            reply = self._query_local_llm(llm_prompt)
            if reply:
                return reply
            dialogues = OFFLINE_DIALOGUES[self.personality].get("fatigue", [])
            return random.choice(dialogues)

        # 3. Contextual category mapping
        category = "code"
        if any(term in app for term in ["code", "xcode", "cursor", "pycharm", "neovim", "vim", "sublime"]):
            category = "code"
        elif any(term in app for term in ["terminal", "ghostty", "iterm", "kitty", "alacritty", "warp"]):
            category = "terminal"
        elif any(term in app for term in ["chrome", "safari", "firefox", "edge", "arc", "brave"]):
            category = "browser"
        elif any(term in app for term in ["slack", "wechat", "weixin", "teams", "lark", "feishu", "dingtalk"]):
            category = "chat"
        else:
            category = "code"

        # Try LLM first
        llm_prompt = f"用户当前正在使用软件 '{context.get('app', '未知')}'，窗口标题是 '{title}'。请结合情境说一句有趣的话。"
        reply = self._query_local_llm(llm_prompt)
        if reply:
            return reply

        # Fallback to offline dialogue pool
        pool = OFFLINE_DIALOGUES[self.personality].get(category, [])
        if not pool:
            pool = OFFLINE_DIALOGUES[self.personality]["code"]

        choices = [c for c in pool if c != self.last_response]
        if not choices:
            choices = pool
        chosen = random.choice(choices)
        self.last_response = chosen
        return chosen

    def chat(self, user_message: str) -> str:
        """Handle direct interactive questions from user in speech bubble."""
        llm_prompt = f"用户对你说了：'{user_message}'，请用你的专属人设直接回答它（25字以内）。"
        reply = self._query_local_llm(llm_prompt)
        if reply:
            return reply

        # Offline fallback responses
        fallback_replies = {
            "clippy": f"收到！关于「{user_message}」，我的建议是：喝杯热咖啡，然后继续干！",
            "cat": f"喵？「{user_message}」？本喵听不懂，除非你拿小鱼干来换回答！",
            "hacker": f"关于「{user_message}」：先看文档，再看源码，一切自然迎刃而解。",
            "zen": f"世间烦恼，皆因执念。面对「{user_message}」，不妨先深吸一口气。",
        }
        return fallback_replies.get(self.personality, "我在听呢！有什么我可以帮你的？")
