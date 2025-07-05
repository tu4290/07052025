# Local LLM API Integration Guide
## Your Personal AI Models - No Rate Limits, Complete Freedom

This guide shows you how to use your local HuiHui-MoE and DeepSeek V2 models in various IDEs and tools.

## 🔑 **Your API Keys**

### **Master Keys**
- **Master Key**: `eots-local-master-2025-elite-options`
- **HuiHui-MoE Key**: `huihui-moe-specialist-expert-system`  
- **DeepSeek V2 Key**: `deepseek-v2-coding-assistant-elite`

### **API Endpoint**
- **Base URL**: `http://localhost:11434`
- **Chat Endpoint**: `http://localhost:11434/api/chat`

## 🛠️ **IDE Integration**

### **VS Code / Cursor Integration**

#### **1. Continue.dev Extension**
```json
// Add to Continue config
{
  "models": [
    {
      "title": "HuiHui-MoE EOTS Expert",
      "provider": "ollama",
      "model": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
      "apiBase": "http://localhost:11434",
      "apiKey": "huihui-moe-specialist-expert-system"
    },
    {
      "title": "DeepSeek V2 Coding Assistant", 
      "provider": "ollama",
      "model": "deepseek-coder-v2:16b",
      "apiBase": "http://localhost:11434",
      "apiKey": "deepseek-v2-coding-assistant-elite"
    }
  ]
}
```

#### **2. GitHub Copilot Alternative**
```json
// settings.json
{
  "continue.telemetryEnabled": false,
  "continue.manuallyTriggerCompletion": false,
  "continue.enableTabAutocomplete": true,
  "continue.models": [
    {
      "title": "DeepSeek V2 Local",
      "provider": "ollama", 
      "model": "deepseek-coder-v2:16b"
    }
  ]
}
```

### **PyCharm / IntelliJ Integration**

#### **1. AI Assistant Plugin**
```yaml
# AI Assistant Configuration
endpoint: "http://localhost:11434/api/chat"
api_key: "deepseek-v2-coding-assistant-elite"
model: "deepseek-coder-v2:16b"
temperature: 0.1
max_tokens: 4000
```

#### **2. Custom Plugin Setup**
```python
# PyCharm plugin configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "models": {
        "coding": {
            "name": "deepseek-coder-v2:16b",
            "api_key": "deepseek-v2-coding-assistant-elite"
        },
        "analysis": {
            "name": "huihui_ai/huihui-moe-abliterated:5b-a1.7b", 
            "api_key": "huihui-moe-specialist-expert-system"
        }
    }
}
```

## 🐍 **Python Integration**

### **Quick Usage**
```python
from huihui_integration.core.local_llm_client import LocalLLMClient

# Initialize client
client = LocalLLMClient()

# Market analysis with HuiHui-MoE
analysis = client.chat_huihui(
    "Analyze SPY options flow and regime", 
    specialist="orchestrator"
)

# Coding help with DeepSeek V2
code = client.chat_deepseek(
    "Optimize this pandas dataframe operation",
    mode="development"
)
```

### **Advanced Usage**
```python
# Specialist routing
regime_analysis = client.chat_huihui(
    "What's the current VIX regime?", 
    specialist="market_regime"
)

flow_analysis = client.chat_huihui(
    "Interpret these VAPI-FA signals", 
    specialist="options_flow"
)

sentiment_analysis = client.chat_huihui(
    "Analyze market sentiment from news", 
    specialist="sentiment"
)

# Final strategic decision
strategy = client.chat_huihui(
    f"Given: {regime_analysis} {flow_analysis} {sentiment_analysis}. What's the optimal strategy?",
    specialist="orchestrator"
)
```

## 🌐 **REST API Usage**

### **cURL Examples**

#### **HuiHui-MoE Market Analysis**
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Authorization: Bearer huihui-moe-specialist-expert-system" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
    "messages": [
      {
        "role": "system", 
        "content": "You are a market regime analysis expert."
      },
      {
        "role": "user", 
        "content": "Analyze current SPY market regime"
      }
    ],
    "stream": false,
    "options": {
      "temperature": 0.1,
      "num_predict": 2000
    }
  }'
```

#### **DeepSeek V2 Coding**
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Authorization: Bearer deepseek-v2-coding-assistant-elite" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2:16b",
    "messages": [
      {
        "role": "user", 
        "content": "Write a Python function for Black-Scholes option pricing"
      }
    ],
    "stream": false,
    "options": {
      "temperature": 0.1,
      "num_predict": 4000
    }
  }'
```

### **JavaScript/Node.js**
```javascript
const axios = require('axios');

async function chatWithHuiHui(prompt, specialist = 'general') {
  const response = await axios.post('http://localhost:11434/api/chat', {
    model: 'huihui_ai/huihui-moe-abliterated:5b-a1.7b',
    messages: [
      { role: 'user', content: prompt }
    ],
    stream: false,
    options: { temperature: 0.1 }
  }, {
    headers: {
      'Authorization': 'Bearer huihui-moe-specialist-expert-system',
      'Content-Type': 'application/json'
    }
  });
  
  return response.data.message.content;
}
```

## 🔧 **Environment Setup**

### **Load Environment Variables**
```bash
# Load your API keys
source .env.local_llm

# Or in PowerShell
Get-Content .env.local_llm | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
  }
}
```

### **HuiHui Local LLM Integration**
```python
# Use HuiHui Local LLM Client (Recommended)
from huihui_integration.core.local_llm_client import LocalLLMClient

client = LocalLLMClient()

# Direct HuiHui expert communication
response = client.chat_huihui(
    prompt="Analyze current market regime for SPY",
    specialist="market_regime",
    temperature=0.1
)
```

## 🎯 **Specialist Routing Guide**

### **HuiHui-MoE Specialists**
- **`market_regime`**: VRI, volatility, regime detection
- **`options_flow`**: VAPI-FA, DWFD, institutional flow
- **`sentiment`**: News analysis, market psychology
- **`orchestrator`**: Strategic synthesis, final decisions

### **DeepSeek V2 Modes**
- **`development`**: General coding assistance
- **`debugging`**: Error analysis and fixes
- **`architecture`**: System design and planning
- **`review`**: Code quality and optimization

## 🚀 **Performance Tips**

1. **Model Loading**: First request loads model (~10-30 seconds)
2. **Concurrent Requests**: Max 4 simultaneous requests
3. **Context Length**: HuiHui (32K), DeepSeek (16K)
4. **Temperature**: 0.1 for focused, 0.3 for creative
5. **Caching**: Models stay loaded for faster subsequent requests

## 🔒 **Security Notes**

- API keys are for local use only
- Server runs on localhost (127.0.0.1)
- No external network access required
- All processing happens on your machine
- Complete privacy and control

---

**🎉 Congratulations! You now have unlimited, private AI assistance with no rate limits!**
