import type { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { ChatOllama } from '@langchain/ollama'
import { ChatOpenAI } from '@langchain/openai'
import { config, requireLlamaBaseUrl, requireOpenAiKey } from './config.js'

// Returns a provider-agnostic chat model so each step is written once and the
// provider is chosen purely by the PROVIDER env var.
export const buildChatModel = (): BaseChatModel => {
  if (config.provider === 'ollama') {
    return new ChatOllama({
      model: config.ollamaModel,
      baseUrl: requireLlamaBaseUrl(),
      temperature: 0,
      format: 'json',
    })
  }

  return new ChatOpenAI({
    apiKey: requireOpenAiKey(),
    model: config.openAiModel,
    temperature: 0,
    timeout: 60_000,
  })
}
