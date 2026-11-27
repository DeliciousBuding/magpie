package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// openAIStreamUsage is the usage chunk OpenAI's Chat API ends a stream
// with when asked for it (stream_options.include_usage): prompt_tokens
// takes in the 1920 read from the cache.
const openAIStreamUsage = `{"prompt_tokens":2006,"completion_tokens":300,"total_tokens":2306,"prompt_tokens_details":{"cached_tokens":1920,"audio_tokens":0},"completion_tokens_details":{"reasoning_tokens":0,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}}`

// promptStreams are Chat streams of one reply to a 2006-token prompt, 1920
// of it read from the cache: OpenAI's, and streams that count the prompt
// on more than one chunk, the cache named on some of them only.
var promptStreams = []struct {
	name   string
	chunks []string
}{
	{"openai", []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}],"usage":null}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[],"usage":` + openAIStreamUsage + `}`,
	}},
	{"cache on the first count only", []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":2006,"completion_tokens":0,"total_tokens":2006,"prompt_tokens_details":{"cached_tokens":1920}}}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2006,"completion_tokens":300,"total_tokens":2306}}`,
	}},
	{"prompt first, cache last", []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":2006,"completion_tokens":0,"total_tokens":2006}}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":300,"total_tokens":300,"prompt_tokens_details":{"cached_tokens":1920}}}`,
	}},
	{"prompt, cache-only, output-only", []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":2006,"completion_tokens":0,"total_tokens":2006}}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":1920}}}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":300,"total_tokens":300}}`,
	}},
}

// chatStreamer answers every request with chunks, as a stream.
func chatStreamer(t *testing.T, chunks []string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			io.WriteString(w, "data: "+c+"\n\n")
		}
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A Responses client asking a Chat provider is told input_tokens as the
// whole prompt, the cache among it, never the prompt and the cache again
// (cyberestro on X: Codex's input_tokens was the prompt and its cache
// together, its cache hit rate under half, and it compacted early). The
// ledger records the prompt the same way: 86 in, 1920 read from cache.
func TestResponsesInputTokensCountTheCacheOnce(t *testing.T) {
	for _, s := range promptStreams {
		for _, stream := range []string{"true", "false"} {
			t.Run(s.name+" stream="+stream, func(t *testing.T) {
				fresh(t)
				up := chatStreamer(t, s.chunks)
				if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"m1"}, Chat: up.URL + "/v1"}); err != nil {
					t.Fatal(err)
				}
				code, reply := post(t, "/v1/responses", `{"model":"fake/m1","input":"hi","stream":`+stream+`}`)
				if code != 200 {
					t.Fatalf("%d %s", code, reply)
				}
				var u struct {
					Input   int `json:"input_tokens"`
					Total   int `json:"total_tokens"`
					Details struct {
						Cached int `json:"cached_tokens"`
					} `json:"input_tokens_details"`
				}
				found := false
				if stream == "true" {
					for _, ev := range events(reply) {
						if ev["type"] == "response.completed" {
							b, _ := json.Marshal(ev["response"].(map[string]any)["usage"])
							found = json.Unmarshal(b, &u) == nil
						}
					}
				} else {
					var whole struct {
						Usage json.RawMessage `json:"usage"`
					}
					found = json.Unmarshal([]byte(reply), &whole) == nil && json.Unmarshal(whole.Usage, &u) == nil
				}
				if !found {
					t.Fatalf("no usage in %s", reply)
				}
				if u.Input != 2006 || u.Details.Cached != 1920 || u.Total != 2306 {
					t.Errorf("input_tokens %d, cached_tokens %d, total_tokens %d; want 2006, 1920, 2306", u.Input, u.Details.Cached, u.Total)
				}
				if r := lastUsage(t); r.Input != 86 || r.CacheRead != 1920 || r.Output != 300 {
					t.Errorf("recorded in %d, cache read %d, out %d; want 86, 1920, 300", r.Input, r.CacheRead, r.Output)
				}
			})
		}
	}
}

// The siblings: a Messages client of the same streams is told 86 input
// beside the 1920 read from cache, and a Chat client's stream passed
// through as it came is recorded the same.
func TestChatStreamPromptCountedOnceEverywhere(t *testing.T) {
	for _, s := range promptStreams {
		t.Run(s.name, func(t *testing.T) {
			fresh(t)
			up := chatStreamer(t, s.chunks)
			if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"m1"}, Chat: up.URL + "/v1"}); err != nil {
				t.Fatal(err)
			}
			code, reply := post(t, "/v1/messages", `{"model":"fake/m1","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
			if code != 200 {
				t.Fatalf("%d %s", code, reply)
			}
			if u := messagesUsage(t, reply); u["input_tokens"] != 86 || u["cache_read_input_tokens"] != 1920 {
				t.Errorf("messages usage %v; want 86 input, 1920 read from cache", u)
			}
			code, reply = post(t, "/v1/chat/completions", `{"model":"fake/m1","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
			if code != 200 {
				t.Fatalf("%d %s", code, reply)
			}
			if r := lastUsage(t); r.Input != 86 || r.CacheRead != 1920 {
				t.Errorf("chat passthrough recorded in %d, cache read %d; want 86, 1920", r.Input, r.CacheRead)
			}
		})
	}
}

// Gemini's usageMetadata comes on every chunk; one naming the cache before
// one that doesn't still tells the prompt once.
func TestGeminiStreamPromptCountedOnce(t *testing.T) {
	out := toAnthropic(t, (&codeAssistDecoder{}).decode,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Read"}]}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":1,"cachedContentTokenCount":40}}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"ing."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10}}`,
	)
	if u := usageOfType(t, out, "message_delta"); u["input_tokens"] != float64(60) || u["cache_read_input_tokens"] != float64(40) {
		t.Errorf("message_delta usage %v; want 60 input, 40 read from cache", u)
	}
	s := newSniffer(provider.Gemini, "text/event-stream")
	s.write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Read\"}]}}],\"usageMetadata\":{\"promptTokenCount\":100,\"candidatesTokenCount\":1,\"cachedContentTokenCount\":40}}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ing.\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":100,\"candidatesTokenCount\":10}}\n\n"))
	if u := s.usage(); u.Input != 60 || u.CacheRead != 40 || u.Output != 10 {
		t.Errorf("sniffed in %d, cache read %d, out %d; want 60, 40, 10", u.Input, u.CacheRead, u.Output)
	}
}

func TestStreamReportsFullCacheHitAtTheEnd(t *testing.T) {
	for _, tc := range []struct {
		name    string
		proto   provider.Protocol
		decode  func(string, func(Event)) error
		chunks  []string
		wantOut int
	}{
		{"gemini", provider.Gemini, (&codeAssistDecoder{}).decode, []string{
			`{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":100}}`,
			`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":100,"candidatesTokenCount":2}}`,
		}, 2},
		{"chat", provider.Chat, (&chatDecoder{}).decode, []string{
			`{"choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":100}}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":100},"completion_tokens":2}}`,
		}, 2},
		{"chat cache-only then output", provider.Chat, (&chatDecoder{}).decode, []string{
			`{"choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":100}}`,
			`{"choices":[{"delta":{}}],"usage":{"prompt_tokens_details":{"cached_tokens":100}}}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}`,
		}, 2},
		{"gemini zero output full cache", provider.Gemini, (&codeAssistDecoder{}).decode, []string{
			`{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":100}}`,
			`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":100}}`,
		}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := toAnthropic(t, tc.decode, tc.chunks...)
			if u := usageOfType(t, out, "message_start"); u["input_tokens"] != float64(100) || u["cache_read_input_tokens"] != float64(0) {
				t.Errorf("start usage %v; want 100 input, 0 cached", u)
			}
			if u := usageOfType(t, out, "message_delta"); u["input_tokens"] != float64(0) || u["cache_read_input_tokens"] != float64(100) || u["output_tokens"] != float64(tc.wantOut) {
				t.Errorf("final usage %v; want 0 input, 100 cached, %d output", u, tc.wantOut)
			}
			s := newSniffer(tc.proto, "text/event-stream")
			for _, chunk := range tc.chunks {
				s.write([]byte("data: " + chunk + "\n\n"))
			}
			billed := s.usage().billed()
			if billed.Input != 0 || billed.CacheRead != 100 || billed.Output != tc.wantOut {
				t.Errorf("billed %d input, %d cached, %d output; want 0, 100, %d", billed.Input, billed.CacheRead, billed.Output, tc.wantOut)
			}
			ru := routeUsage("fake", "m1", billed)
			if len(ru) != 1 || ru[0].Input != 0 || ru[0].CacheRead != 100 || ru[0].Output != tc.wantOut {
				t.Errorf("routeUsage %+v; want 0 input, 100 cached, %d output", ru, tc.wantOut)
			}
		})
	}
	// An output-only tail chunk after an Anthropic-named cache update keeps
	// the uncached input instead of reapplying an earlier prompt count.
	out := toAnthropic(t, (&chatDecoder{}).decode,
		`{"choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":100}}`,
		`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":40,"cache_read_input_tokens":100}}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5}}`,
	)
	if u := usageOfType(t, out, "message_delta"); u["input_tokens"] != float64(40) || u["cache_read_input_tokens"] != float64(100) || u["output_tokens"] != float64(5) {
		t.Errorf("anthropic-named then output-only: %v; want 40 input, 100 cached, 5 output", u)
	}
}
