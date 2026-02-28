import { describe, test, expect, afterEach } from "bun:test"
import path from "path"
import os from "os"
import fs from "fs/promises"
import type { LanguageModelV2Prompt } from "@ai-sdk/provider"

import { Config } from "../../src/config/config"
import { Instance } from "../../src/project/instance"
import { Provider } from "../../src/provider/provider"
import { tmpdir } from "../fixture/fixture"

const TEST_PROMPT: LanguageModelV2Prompt = [{ role: "user", content: [{ type: "text", text: "hi" }] }]

async function writeScript(content: string): Promise<string> {
  const p = path.join(os.tmpdir(), `test-key-helper-${Math.random().toString(36).slice(2)}.sh`)
  await fs.writeFile(p, content, { mode: 0o755 })
  return p
}

/**
 * Calls doStream on the language model just enough to trigger the fetch wrapper.
 * The stub response will cause SSE parsing to fail, which is expected and caught.
 */
async function triggerFetch(lang: any): Promise<void> {
  try {
    const result = await lang.doStream({ prompt: TEST_PROMPT, includeRawChunks: false })
    const reader = result.stream.getReader()
    while (true) {
      const { done } = await reader.read().catch(() => ({ done: true, value: undefined }))
      if (done) break
    }
  } catch {
    // Expected — stub response won't parse as Anthropic SSE. The fetch was still called.
  }
}

/** Replaces globalThis.fetch with a capture mock and returns headers from the first call. */
function mockFetch(): { captured: { headers: Headers | null }; restore: () => void } {
  const captured = { headers: null as Headers | null }
  const original = globalThis.fetch
  globalThis.fetch = (async (_input: any, init: any) => {
    if (captured.headers === null) {
      captured.headers = new Headers(init?.headers ?? {})
    }
    return new Response("", { status: 200 })
  }) as typeof fetch
  return { captured, restore: () => { globalThis.fetch = original } }
}

// ── Config schema tests ───────────────────────────────────────────────────────

describe("Config.Provider options schema — apiKeyHelper fields", () => {
  test("accepts apiKeyHelper as an absolute path string", () => {
    const result = Config.Provider.safeParse({
      options: { apiKeyHelper: "/usr/local/bin/get-token.sh" },
    })
    expect(result.success).toBe(true)
    expect(result.data?.options?.apiKeyHelper).toBe("/usr/local/bin/get-token.sh")
  })

  test("accepts apiKeyHelper as a home-relative path", () => {
    const result = Config.Provider.safeParse({
      options: { apiKeyHelper: "~/bin/get-token.sh" },
    })
    expect(result.success).toBe(true)
    expect(result.data?.options?.apiKeyHelper).toBe("~/bin/get-token.sh")
  })

  test("accepts apiKeyHelperTTL as a positive integer", () => {
    const result = Config.Provider.safeParse({
      options: { apiKeyHelperTTL: 1_800_000 },
    })
    expect(result.success).toBe(true)
    expect(result.data?.options?.apiKeyHelperTTL).toBe(1_800_000)
  })

  test("rejects apiKeyHelperTTL = 0 (must be positive)", () => {
    const result = Config.Provider.safeParse({
      options: { apiKeyHelperTTL: 0 },
    })
    expect(result.success).toBe(false)
  })

  test("rejects apiKeyHelperTTL as a float", () => {
    const result = Config.Provider.safeParse({
      options: { apiKeyHelperTTL: 3600.5 },
    })
    expect(result.success).toBe(false)
  })

  test("accepts customAuthHeaders as a string array", () => {
    const result = Config.Provider.safeParse({
      options: { customAuthHeaders: ["X-Api-Key", "X-Token"] },
    })
    expect(result.success).toBe(true)
    expect(result.data?.options?.customAuthHeaders).toEqual(["X-Api-Key", "X-Token"])
  })

  test("accepts an empty customAuthHeaders array", () => {
    const result = Config.Provider.safeParse({
      options: { customAuthHeaders: [] },
    })
    expect(result.success).toBe(true)
  })

  test("accepts customHeaders as a string-to-string record", () => {
    const result = Config.Provider.safeParse({
      options: { customHeaders: { "X-Team": "engineering", "X-Env": "prod" } },
    })
    expect(result.success).toBe(true)
    expect(result.data?.options?.customHeaders).toEqual({ "X-Team": "engineering", "X-Env": "prod" })
  })

  test("all four new fields are optional", () => {
    const result = Config.Provider.safeParse({ options: { apiKey: "k" } })
    expect(result.success).toBe(true)
    expect(result.data?.options?.apiKeyHelper).toBeUndefined()
    expect(result.data?.options?.apiKeyHelperTTL).toBeUndefined()
    expect(result.data?.options?.customAuthHeaders).toBeUndefined()
    expect(result.data?.options?.customHeaders).toBeUndefined()
  })

  test("all four fields coexist correctly", () => {
    const result = Config.Provider.safeParse({
      options: {
        apiKey: "key",
        apiKeyHelper: "~/bin/tok.sh",
        apiKeyHelperTTL: 1_800_000,
        customAuthHeaders: ["X-Api-Key"],
        customHeaders: { "X-Team": "eng" },
      },
    })
    expect(result.success).toBe(true)
    const opts = result.data?.options
    expect(opts?.apiKeyHelper).toBe("~/bin/tok.sh")
    expect(opts?.apiKeyHelperTTL).toBe(1_800_000)
    expect(opts?.customAuthHeaders).toEqual(["X-Api-Key"])
    expect(opts?.customHeaders).toEqual({ "X-Team": "eng" })
  })
})

// ── Path expansion tests ──────────────────────────────────────────────────────

describe("home-relative path expansion", () => {
  test("expands ~/... to an absolute path under the home directory", () => {
    const script = "~/bin/get-token.sh"
    const expanded = script.startsWith("~/")
      ? path.join(os.homedir(), script.slice(2))
      : script
    expect(expanded).toBe(path.join(os.homedir(), "bin/get-token.sh"))
    expect(expanded.startsWith("~")).toBe(false)
    expect(path.isAbsolute(expanded)).toBe(true)
  })

  test("absolute path passes through unchanged", () => {
    const script = "/usr/local/bin/get-token.sh"
    const expanded = script.startsWith("~/")
      ? path.join(os.homedir(), script.slice(2))
      : script
    expect(expanded).toBe("/usr/local/bin/get-token.sh")
  })
})

// ── Fetch integration tests ───────────────────────────────────────────────────

describe("apiKeyHelper fetch integration", () => {
  const scripts: string[] = []
  afterEach(async () => {
    for (const s of scripts.splice(0)) {
      await fs.rm(s, { force: true })
      await fs.rm(s + ".count", { force: true }) // counter file used by TTL test scripts
    }
  })

  async function helper(content: string): Promise<string> {
    const p = await writeScript(content)
    scripts.push(p)
    return p
  }

  test("injects Authorization: Bearer header from script stdout", async () => {
    const script = await helper("#!/bin/sh\necho 'my-token-abc'")
    const { captured, restore } = mockFetch()

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: { apiKey: "dummy", apiKeyHelper: script },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await triggerFetch(lang)
          expect(captured.headers).not.toBeNull()
          expect(captured.headers?.get("authorization")).toBe("Bearer my-token-abc")
        },
      })
    } finally {
      restore()
    }
  })

  test("trims leading/trailing whitespace from script output", async () => {
    const script = await helper("#!/bin/sh\nprintf '  trimmed-token  '")
    const { captured, restore } = mockFetch()

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: { apiKey: "dummy", apiKeyHelper: script },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await triggerFetch(lang)
          expect(captured.headers?.get("authorization")).toBe("Bearer trimmed-token")
        },
      })
    } finally {
      restore()
    }
  })

  test("injects customAuthHeaders alongside Authorization", async () => {
    const script = await helper("#!/bin/sh\necho 'shared-token'")
    const { captured, restore } = mockFetch()

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: {
                apiKey: "dummy",
                apiKeyHelper: script,
                customAuthHeaders: ["X-Api-Key", "X-Token"],
              },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await triggerFetch(lang)
          expect(captured.headers?.get("authorization")).toBe("Bearer shared-token")
          expect(captured.headers?.get("x-api-key")).toBe("shared-token")
          expect(captured.headers?.get("x-token")).toBe("shared-token")
        },
      })
    } finally {
      restore()
    }
  })

  test("injects static customHeaders without apiKeyHelper", async () => {
    const { captured, restore } = mockFetch()

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: {
                apiKey: "dummy",
                customHeaders: { "X-Team": "engineering", "X-Env": "prod" },
              },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await triggerFetch(lang)
          expect(captured.headers?.get("x-team")).toBe("engineering")
          expect(captured.headers?.get("x-env")).toBe("prod")
        },
      })
    } finally {
      restore()
    }
  })

  test("TTL caching: script is not re-run within the TTL window", async () => {
    // Script increments a counter file on each real execution; cached calls won't re-run it
    const script = await helper(
      `#!/bin/sh\nCOUNT="\${0}.count"\nN=$(cat "$COUNT" 2>/dev/null || echo 0)\nN=$((N + 1))\necho "$N" > "$COUNT"\necho "token-$N"`,
    )
    const authValues: Array<string | null> = []
    const original = globalThis.fetch
    globalThis.fetch = (async (_input: any, init: any) => {
      authValues.push(new Headers(init?.headers ?? {}).get("authorization"))
      return new Response("", { status: 200 })
    }) as typeof fetch

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: {
                apiKey: "dummy",
                apiKeyHelper: script,
                apiKeyHelperTTL: 60_000, // 1 min — both calls land well within window
              },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await triggerFetch(lang)
          await triggerFetch(lang)
          expect(authValues).toHaveLength(2)
          expect(authValues[0]).toBe(authValues[1])
        },
      })
    } finally {
      globalThis.fetch = original
    }
  })

  test("TTL caching: script is re-run after the TTL expires", async () => {
    const script = await helper(
      `#!/bin/sh\nCOUNT="\${0}.count"\nN=$(cat "$COUNT" 2>/dev/null || echo 0)\nN=$((N + 1))\necho "$N" > "$COUNT"\necho "token-$N"`,
    )
    const authValues: Array<string | null> = []
    const original = globalThis.fetch
    globalThis.fetch = (async (_input: any, init: any) => {
      authValues.push(new Headers(init?.headers ?? {}).get("authorization"))
      return new Response("", { status: 200 })
    }) as typeof fetch

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: {
                apiKey: "dummy",
                apiKeyHelper: script,
                apiKeyHelperTTL: 50, // 50 ms — expires before second call
              },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await triggerFetch(lang)
          await Bun.sleep(100) // Wait for TTL to expire
          await triggerFetch(lang)
          expect(authValues).toHaveLength(2)
          expect(authValues[0]).not.toBe(authValues[1])
        },
      })
    } finally {
      globalThis.fetch = original
    }
  })

  test("empty script output rejects with a descriptive error", async () => {
    const script = await helper("#!/bin/sh\nprintf ''")
    const { restore } = mockFetch()

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: { apiKey: "dummy", apiKeyHelper: script },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await expect(
            lang.doStream({ prompt: TEST_PROMPT, includeRawChunks: false } as any),
          ).rejects.toThrow("apiKeyHelper returned empty output")
        },
      })
    } finally {
      restore()
    }
  })

  test("non-zero script exit code rejects with an error", async () => {
    const script = await helper("#!/bin/sh\nexit 1")
    const { restore } = mockFetch()

    try {
      await using tmp = await tmpdir({
        config: {
          provider: {
            anthropic: {
              options: { apiKey: "dummy", apiKeyHelper: script },
            },
          },
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          const model = Object.values(providers["anthropic"].models)[0]
          const lang = await Provider.getLanguage(model)
          await expect(
            lang.doStream({ prompt: TEST_PROMPT, includeRawChunks: false } as any),
          ).rejects.toThrow()
        },
      })
    } finally {
      restore()
    }
  })
})
